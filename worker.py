"""
UHN AI-ICF background worker.

Runs as the container command for the event-driven Container Apps *job*
``ca-uhn-aiicf-worker``. KEDA starts one job execution per message in the
``icf-jobs`` queue (up to max-executions), and each execution runs this script
once: it receives a single message, runs the RLM pipeline, writes the outputs
to blob storage, emails the user, updates the job row, deletes the message,
and exits.

KEDA only *triggers* executions based on queue length — it does not hand the
message to the container. This worker is responsible for receiving and deleting
the message itself (see receive_message / delete_message in icf.jobs).
"""

from __future__ import annotations

import base64
import html as html_module
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("icf.worker")

from icf import jobs  # noqa: E402
from icf.pipeline import ICFPipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent

# A message reprocessed more than this many times (e.g. it keeps crashing the
# worker) is treated as poison: marked failed and dropped instead of looping.
MAX_DEQUEUE = int(os.environ.get("ICF_MAX_DEQUEUE", "3"))
# Must be >= the message visibility timeout used when receiving, and <= the
# job's --replica-timeout. A single ICF run is ~20-30 min; this leaves headroom.
VISIBILITY_TIMEOUT = int(os.environ.get("ICF_VISIBILITY_TIMEOUT", "5400"))

ACS_CONNECTION_STRING = os.environ.get("ACS_CONNECTION_STRING")
ACS_SENDER_ADDRESS = os.environ.get("ACS_SENDER_ADDRESS")
# Friendly name shown to recipients (set MailFrom display name in ACS portal too).
ACS_SENDER_DISPLAY_NAME = os.environ.get("ACS_SENDER_NAME", "UHN AI-Hub")
# Where replies are routed (must be a real UHN mailbox).
ACS_REPLY_TO_ADDRESS = os.environ.get("ACS_REPLY_TO", "AIHub@uhn.ca")
ACS_REPLY_TO_NAME = os.environ.get("ACS_REPLY_TO_NAME", ACS_SENDER_DISPLAY_NAME)
ICF_SHAREPOINT_URL = os.environ.get(
    "ICF_SHAREPOINT_URL",
    "https://universityhealthnetwork.sharepoint.com/sites/AIHub/SitePages/AI-ICF.aspx",
)
SUPPORT_EMAIL = os.environ.get("ICF_SUPPORT_EMAIL", "AIHub@uhn.ca")
EMAIL_ENABLED = bool(ACS_CONNECTION_STRING and ACS_SENDER_ADDRESS)


def _email_content(protocol_name: str) -> dict[str, str]:
    """Build subject + plain/HTML bodies for the completion email.

    Formatting mirrors the approved Email_Template_Wording.docx: bold labels,
    an italicized protocol name, bold section headers with extra spacing,
    real bullet lists for the attachment/reminder items, and a bold
    "Need Help?" / "We'd Appreciate Your Feedback" heading with normal body text.
    """
    subject = "Your AI-ICF Draft is ready for your review"
    protocol_safe = html_module.escape(protocol_name)

    plain_text = f"""Hello,

Your AI-generated draft Informed Consent Form (ICF) is attached and ready for review.

Source Protocol: {protocol_name}

INCLUDED ATTACHMENTS

• Draft Version: This is your working draft consent form. Please review, edit, and refine the content as needed before submission through CAPCR.

• Marked-Up Version: This version shows where AI-generated content came from in the study protocol. Use the references and comments in this document to quickly trace content back to the protocol and better understand how the draft was generated.

IMPORTANT REMINDERS:

• This is an AI-generated draft intended to support the consent form development process. All content must be carefully reviewed and validated by the study team before submission into CAPCR for REB approval.

• Ensure all instructional and placeholder text (highlighted text and text in grey italics) is removed before submitting into CAPCR for REB approval.

NEED HELP?
Visit the AI-ICF SharePoint page ({ICF_SHAREPOINT_URL}) for guidance, FAQs, and additional resources, or contact {SUPPORT_EMAIL} if you have questions or need support.

WE'D APPRECIATE YOUR FEEDBACK:
After submitting your study in CAPCR, you will receive a brief evaluation survey by email. Your feedback is invaluable and helps us improve the AI-ICF tool and inform future enhancements. We encourage you to complete the survey when it arrives.

Thank you for using the AI-ICF Tool.

AI-ICF Tool Project Team
"""

    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"></head>
<body style="font-family: Calibri, Arial, Helvetica, sans-serif; font-size: 15px; line-height: 1.6; color: #212529; margin: 0; padding: 0;">
  <p style="margin: 0 0 16px 0;">Hello,</p>

  <p style="margin: 0 0 16px 0;">Your AI-generated draft Informed Consent Form (ICF) is attached and ready for review.</p>

  <p style="margin: 0 0 16px 0;"><strong>Source Protocol:</strong> <em>{protocol_safe}</em></p>

  <p style="margin: 24px 0 8px 0; font-weight: bold;">Included Attachments</p>

  <ul style="margin: 0 0 16px 0; padding-left: 22px;">
    <li style="margin-bottom: 12px;"><strong>Draft Version:</strong> This is your working draft consent form. Please review, edit, and refine the content as needed before submission through CAPCR.</li>
    <li style="margin-bottom: 0;"><strong>Marked-Up Version:</strong> This version shows where AI-generated content came from in the study protocol. Use the references and comments in this document to quickly trace content back to the protocol and better understand how the draft was generated.</li>
  </ul>

  <p style="margin: 24px 0 8px 0; font-weight: bold;">Important Reminders:</p>

  <ul style="margin: 0 0 16px 0; padding-left: 22px;">
    <li style="margin-bottom: 12px;">This is an AI-generated draft intended to support the consent form development process. All content must be carefully reviewed and validated by the study team before submission into CAPCR for REB approval.</li>
    <li style="margin-bottom: 0;">Ensure all instructional and placeholder text (highlighted text and text in grey italics) is removed before submitting into CAPCR for REB approval.</li>
  </ul>

  <p style="margin: 24px 0 24px 0; font-weight: bold;">Need Help? Visit the <a href="{ICF_SHAREPOINT_URL}" style="color: #0563C1; font-weight: bold;">AI-ICF SharePoint page</a> for guidance, FAQs, and additional resources, or contact <a href="mailto:{SUPPORT_EMAIL}" style="color: #0563C1; font-weight: bold;">{SUPPORT_EMAIL}</a> if you have questions or need support.</p>

  <p style="margin: 0 0 16px 0;"><strong>We'd Appreciate Your Feedback:</strong> After submitting your study in CAPCR, you will receive a brief evaluation survey by email. Your feedback is invaluable and helps us improve the AI-ICF tool and inform future enhancements. We encourage you to complete the survey when it arrives.</p>

  <p style="margin: 0 0 16px 0;">Thank you for using the AI-ICF Tool.</p>

  <p style="margin: 0;"><em>AI-ICF Tool Project Team</em></p>
</body>
</html>"""

    return {"subject": subject, "plainText": plain_text, "html": html_body}


def send_output_email(
    to_email: str,
    outputs: list[tuple[str, bytes, str]],
    protocol_name: str,
) -> tuple[bool, str]:
    """Email the draft + marked-up ICFs as attachments. Returns (ok, message)."""
    if not EMAIL_ENABLED:
        return False, "Email delivery is not configured on this deployment."

    attachments = [
        {
            "name": name,
            "contentType": mime,
            "contentInBase64": base64.b64encode(data).decode("utf-8"),
        }
        for name, data, mime in outputs
        if name.startswith(("draft_icf_", "marked_up_icf_"))
    ]
    if not attachments:
        return False, "No ICF documents were produced to email."

    try:
        from azure.communication.email import EmailClient

        client = EmailClient.from_connection_string(ACS_CONNECTION_STRING)
        bodies = _email_content(protocol_name)
        message = {
            "senderAddress": ACS_SENDER_ADDRESS,
            "recipients": {"to": [{"address": to_email, "displayName": to_email}]},
            "replyTo": [
                {
                    "address": ACS_REPLY_TO_ADDRESS,
                    "displayName": ACS_REPLY_TO_NAME,
                }
            ],
            "content": {
                "subject": bodies["subject"],
                "plainText": bodies["plainText"],
                "html": bodies["html"],
            },
            "attachments": attachments,
        }
        poller = client.begin_send(message)
        result = poller.result()
        status = str((result or {}).get("status", "Unknown"))
        if status.lower() == "succeeded":
            return True, f"A copy was emailed to {to_email}."
        return False, f"Email send finished with status: {status}."
    except Exception as ex:  # noqa: BLE001
        return False, f"Could not send email: {type(ex).__name__}: {ex}"


def _registry_path(study_label: str) -> str:
    filename = jobs.REGISTRY_FILES.get(study_label)
    if not filename:
        raise ValueError(f"Unknown study label: {study_label!r}")
    return str(REPO_ROOT / "data" / filename)


def _run_pipeline(job: dict, workdir: Path) -> tuple[list[tuple[str, bytes, str]], dict | None]:
    """Download the protocol, run the pipeline, return (outputs, summary)."""
    protocol_name, protocol_bytes = jobs.download_input(job["input_blob"])
    protocol_path = workdir / protocol_name
    protocol_path.write_bytes(protocol_bytes)

    out_dir = workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ICFPipeline(
        protocol_path=str(protocol_path),
        template_path=_registry_path(job["study_label"]),
        output_dir=str(out_dir),
        model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        backend="azure_openai",
        backend_kwargs={
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        },
        extraction_backend="hybrid",
        verbose=False,
        skip_review=False,
        us_funded=bool(job.get("us_funded", False)),
        sdm=bool(job.get("sdm", False)),
        debug_log_dir=None,
    )
    result = pipeline.run()

    outputs: list[tuple[str, bytes, str]] = []
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            outputs.append((f.name, f.read_bytes(), jobs.mime_for(f.suffix)))
    summary = getattr(result, "summary", None)
    return outputs, summary


def process_one() -> bool:
    """Process a single queued job. Returns True if a message was handled."""
    msg, job = jobs.receive_message(visibility_timeout=VISIBILITY_TIMEOUT)
    if msg is None:
        log.info("Queue empty; nothing to do.")
        return False

    owner = (job or {}).get("owner")
    job_id = (job or {}).get("job_id")

    # Malformed / unparseable message: drop it so it can't loop forever.
    if not owner or not job_id:
        log.error("Malformed message (no owner/job_id); deleting.")
        jobs.delete_message(msg)
        return True

    # Poison guard: too many redeliveries means this message keeps killing the
    # worker. Fail the job rather than retry indefinitely.
    if getattr(msg, "dequeue_count", 1) > MAX_DEQUEUE:
        log.error("Job %s exceeded MAX_DEQUEUE (%s); poisoning.", job_id, MAX_DEQUEUE)
        jobs.fail_job(owner, job_id, "Repeatedly failed processing (poison).")
        jobs.delete_message(msg)
        return True

    # Idempotency: a redelivered message for an already-finished job is a no-op.
    existing = jobs.get_job(owner, job_id)
    if existing and existing.get("status") in ("complete", "failed"):
        log.info("Job %s already %s; deleting redelivered message.", job_id, existing["status"])
        jobs.delete_message(msg)
        return True

    workdir = Path(tempfile.mkdtemp(prefix="icfrun_"))
    try:
        log.info("Processing job %s for %s", job_id, owner)
        jobs.mark_running(owner, job_id)

        outputs, summary = _run_pipeline(job, workdir)
        jobs.complete_job(owner, job_id, outputs, summary)
        log.info("Job %s complete (%d files).", job_id, len(outputs))

        if EMAIL_ENABLED and owner and "@" in owner:
            ok, message = send_output_email(owner, outputs, job.get("protocol_name", ""))
            jobs.set_email_status(owner, job_id, ok, message)
            log.info("Email for job %s: ok=%s (%s)", job_id, ok, message)

        # Success -> remove the message so it is not reprocessed.
        jobs.delete_message(msg)
        return True

    except Exception as ex:  # noqa: BLE001
        err = f"{type(ex).__name__}: {ex}\n\n{traceback.format_exc()}"
        log.error("Job %s failed: %s", job_id, err)
        # Deterministic failure (e.g. unreadable protocol): mark failed and delete
        # so it doesn't loop. Infra crashes never reach here — those leave the
        # message to reappear after the visibility timeout and retry.
        jobs.fail_job(owner, job_id, err)
        jobs.delete_message(msg)
        return True
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    jobs.ensure_infra()
    handled = process_one()
    # One message per execution is the canonical event-driven-job contract.
    # Exit 0 either way; KEDA starts a fresh execution for the next message.
    log.info("Worker execution finished (handled=%s).", handled)


if __name__ == "__main__":
    main()