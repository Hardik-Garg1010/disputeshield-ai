"""Create a small PDF evidence packet suitable for Razorpay Documents API."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

def build_packet(dispute: dict, run, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dispute['dispute_id']}_evidence_packet.pdf"
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.leading = 14
    story = [
        Paragraph("DisputeShield Evidence Packet", styles["Title"]),
        Paragraph(f"Dispute: {dispute['dispute_id']} | Amount: {dispute['amount']} {dispute.get('currency','INR')}", body),
        Paragraph(f"Reason: {dispute['reason_code']}", body),
        Spacer(1, 12),
        Paragraph("Evidence summary", styles["Heading2"]),
    ]
    rows = [["Evidence", "Present", "Source"]]
    ev = run.evidence
    for key, label, source_key in [
        ("delivery_confirmed", "Delivery confirmed", "order"),
        ("tracking_available", "Tracking available", "tracking"),
        ("customer_acknowledged_delivery", "Customer acknowledgement", "chat"),
        ("delivery_signature", "Recipient signature", "tracking"),
    ]:
        rows.append([label, "YES" if ev.get(key) else "NO", ev.get("sources", {}).get(source_key, "-")])
    table = Table(rows, colWidths=[150, 70, 290])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeeee")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))
    story.append(table)
    story += [
        Spacer(1, 12),
        Paragraph("AI analysis", styles["Heading2"]),
        Paragraph(f"Contestability score: {run.contestability_score:.3f}", body),
        Paragraph(f"Policy decision: {run.final_decision or run.policy_result.get('decision','')}", body),
        Spacer(1, 8),
        Paragraph("Grounded rebuttal", styles["Heading2"]),
    ]
    rebuttal = run.rebuttal.get("body", "") if isinstance(run.rebuttal, dict) else ""
    for line in rebuttal.splitlines():
        if line.strip():
            story.append(Paragraph(line.replace("&", "&amp;"), body))
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36).build(story)
    return path
