from streamlit.testing.v1 import AppTest


# Test function extract_video_id
def test_extract_video_id():
    at = AppTest.from_file('main.py', default_timeout=30)
    at.run()
    at.text_input[0].input('https://www.youtube.com/watch?v=wDmPgXhlDIg').run()
    at.button[0].click().run()
    assert at.session_state["start"] == True


# Test bad url input
def test_bad_url_input():
    at = AppTest.from_file('main.py', default_timeout=30)
    at.run()
    at.text_input[0].input('123').run()
    assert at.session_state["start"] == False

