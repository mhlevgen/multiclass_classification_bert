from app import app


def test_index_route():
    response = app.test_client().post('/get-category', json={'main_text': 'f', 'add_text': 'f', 'manufacturer': 'j'})

    assert response.status_code == 200
    assert len(response.json) == 4
