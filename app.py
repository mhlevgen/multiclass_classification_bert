import os
import pickle

from flask import Flask, request

from src.preprocessing import remove_punctuation, concat_texts


app = Flask(__name__)
dir_path = os.path.dirname(__file__)

with open(os.path.join(dir_path, 'src/models/clf.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(dir_path, 'src/models/tf_idf.pkl'), 'rb') as f:
    tf_idf = pickle.load(f)

with open(os.path.join(dir_path, 'src/models/ind_to_category.pkl'), 'rb') as f:
    ind_to_category = pickle.load(f)


def process_input(main_text: str, add_text: str, manufacturer: str) -> list[float]:
    main_text = remove_punctuation(main_text)
    add_text = remove_punctuation(add_text)
    manufacturer = remove_punctuation(manufacturer)
    texts = concat_texts(main_text=main_text, add_text=add_text, manufacturer=manufacturer)
    return tf_idf.transform([texts])


@app.route('/get-category', methods=['POST'])
def query_example():
    request_data = request.get_json()

    main_text = request_data['main_text']
    add_text = request_data['add_text']
    manufacturer = request_data['manufacturer']

    input_tf_idf = process_input(main_text, add_text, manufacturer)
    predict = model.predict_proba(input_tf_idf)

    return {cat: predict[0][ind] for ind, cat in ind_to_category.items()}


if __name__ == '__main__':
    app.run(debug=True, port=8889, host='0.0.0.0')
