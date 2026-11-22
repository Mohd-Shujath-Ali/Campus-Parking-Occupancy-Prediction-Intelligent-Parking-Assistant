from recommendation.parking_ranker import ParkingRanker


def _preds(ranker, occ):
    return {z: {"predicted_count": int(m["operational_capacity"] * occ.get(z, 0.5)),
                "saturation_prob": 0.0, "camera_age_seconds": 10.0}
            for z, m in ranker.zones.items()}


def test_clear_winner_is_not_dispersed():
    r = ParkingRanker(seed=1)
    base = r.rank_zones("Block_1", _preds(r, {"ZONE_BLOCK1": 0.05}), apply_anti_herding=False)
    for seed in range(20):
        out = ParkingRanker(seed=seed).rank_zones("Block_1", _preds(r, {"ZONE_BLOCK1": 0.05}))
        assert out[0]["zone_id"] == base[0]["zone_id"]


def test_near_ties_are_dispersed_across_lots():
    r = ParkingRanker(seed=0)
    preds = _preds(r, {})  # all zones 50% full -> scores differ only by distance
    # Block 26's two lots are 50 m and 60 m away: a genuine near-tie
    plain = r.rank_zones("Block_26_CSE", preds, apply_anti_herding=False)
    gap = plain[0]["score"] - plain[1]["score"]
    assert gap <= r.weights["anti_herding_score_delta_threshold"]
    leads = {ParkingRanker(seed=s).rank_zones("Block_26_CSE", preds)[0]["zone_id"] for s in range(60)}
    assert len(leads) >= 2
