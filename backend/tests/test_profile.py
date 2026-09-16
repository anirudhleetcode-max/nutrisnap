PROFILE = {"age": 22, "sex": "female", "height_cm": 162, "weight_kg": 58, "activity": "light", "goal": "lose"}


def test_profile_roundtrip(client, new_user):
    assert client.get("/api/profile", headers=new_user).json()["profile"] is None
    est = client.post("/api/profile/estimate", json=PROFILE, headers=new_user).json()["targets"]
    r = client.put("/api/profile", json=PROFILE, headers=new_user)
    assert r.status_code == 200
    body = client.get("/api/profile", headers=new_user).json()
    assert body["profile"] == PROFILE and body["targets"] == est
    assert "10 x 58 kg" in body["targets"]["breakdown"]["bmr_formula"]
    assert client.get("/api/auth/me", headers=new_user).json()["has_profile"] is True


def test_profile_validation(client, new_user):
    r = client.put("/api/profile", json={**PROFILE, "age": 7}, headers=new_user)
    assert r.status_code == 422
    assert client.put("/api/profile", json=PROFILE).status_code == 401
