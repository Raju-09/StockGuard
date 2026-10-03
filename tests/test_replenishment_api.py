from app.replenishment_api import ReplenishmentAPIRouter

def test_replenishment_api():
    rec = ReplenishmentAPIRouter.get_recommendation("SKU_99", current_stock=10.0, reorder_point=25.0, batch_size=50.0)
    assert rec["order_recommended"] is True
    assert rec["recommended_quantity"] == 50.0
