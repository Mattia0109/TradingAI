from portfolio.portfolio_manager import analyze_portfolio


if __name__ == "__main__":

    print("=" * 60)
    print("TRADING AI - PORTFOLIO ANALYSIS")
    print("=" * 60)

    results = analyze_portfolio()

    print()

    for position, asset in enumerate(results, start=1):

        print(
            f"{position:2d}. "
            f"{asset['ticker']:5} | "
            f"Score: {asset['score']:3} | "
            f"Trend: {asset['trend']:9} | "
            f"Risk: {asset['risk']:6} | "
            f"Decision: {asset['decision']}"
        )