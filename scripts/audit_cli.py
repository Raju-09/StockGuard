import argparse
import sys
from src.inventory.safety_stock import SafetyStockCalculator, StockParameters

def main():
    parser = argparse.ArgumentParser(description="StockGuard CLI Inventory Audit Tool")
    parser.add_argument("--demand", type=float, required=True, help="Average daily demand")
    parser.add_argument("--std-demand", type=float, default=10.0, help="Standard deviation of daily demand")
    parser.add_argument("--lead-time", type=float, required=True, help="Average lead time in days")
    parser.add_argument("--std-lead-time", type=float, default=1.0, help="Standard deviation of lead time")
    parser.add_argument("--service-level", type=float, default=0.95, help="Target service level (e.g. 0.95)")

    args = parser.parse_args()
    params = StockParameters(
        avg_daily_demand=args.demand,
        std_daily_demand=args.std_demand,
        avg_lead_time_days=args.lead_time,
        std_lead_time_days=args.std_lead_time,
        service_level=args.service_level
    )

    ss = SafetyStockCalculator.calculate_safety_stock(params)
    rop = SafetyStockCalculator.calculate_reorder_point(params)

    print("=" * 40)
    print("      STOCKGUARD INVENTORY AUDIT      ")
    print("=" * 40)
    print(f"Target Service Level : {args.service_level * 100:.1f}%")
    print(f"Recommended Safety Stock : {ss} units")
    print(f"Reorder Point (ROP)      : {rop} units")
    print("=" * 40)

if __name__ == "__main__":
    main()
