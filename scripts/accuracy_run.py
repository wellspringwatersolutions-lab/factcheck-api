"""Run the 12 known-answer claims. Abort once the run costs more than $0.30."""

from factcheck.accuracy import CAP_USD, run_accuracy
from factcheck.config import get_settings
from factcheck.wiring import build_service


def main() -> None:
    settings = get_settings()
    service = build_service(settings)
    record = run_accuracy(service, settings, cap=CAP_USD)
    print(
        f"accuracy={record['accuracy']:.1%} checked={record['checked']} "
        f"cost=${record['cost_usd']:.4f} aborted={record['aborted']}"
    )


if __name__ == "__main__":
    main()
