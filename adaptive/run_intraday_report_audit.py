"""CLI per l'audit riproducibile dei report descrittivi intraday."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adaptive.intraday_report_audit import (
    REPORT_MANIFEST_FILENAME,
    STANDARD_INTRADAY_REPORT_FILENAMES,
    audit_intraday_report_directory,
    write_intraday_report_manifest,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verifica completezza, hash, schemi e natura descrittiva dei "
            "CSV intraday. Non calcola segnali, previsioni o P&L."
        )
    )
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Manifest CSV; predefinito: "
            "REPORT_DIR/intraday_report_manifest.csv."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    root = Path(arguments.report_dir)
    report = audit_intraday_report_directory(
        root,
        expected_reports=STANDARD_INTRADAY_REPORT_FILENAMES,
    )
    output = arguments.output or root / REPORT_MANIFEST_FILENAME
    destination = write_intraday_report_manifest(report, output)

    print("\nTRADINGAI INTRADAY REPORT AUDIT — DESCRIPTIVE ONLY")
    print(f"Directory:                 {root}")
    print(
        "Report attesi/osservati: "
        f"{report.observed_expected_reports}/{report.expected_reports}"
    )
    print(f"CSV inattesi:              {report.unexpected_reports}")
    print(f"Righe descrittive:         {report.total_rows}")
    print(f"Esito:                     {report.overall_state}")
    print(f"Manifest:                  {destination}")
    failed = report.details.loc[report.details["audit_state"].ne("PASS")]
    if not failed.empty:
        print("\nELEMENTI DA VERIFICARE")
        for row in failed.itertuples(index=False):
            print(f"- {row.report_name}: {row.audit_state}")
    print(
        "L'audit verifica gli artefatti; non misura accuratezza, rendimento "
        "o idoneita' operativa."
    )
    return 0 if report.overall_state == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
