from streamlit.testing.v1 import AppTest


# Just a simple test
def test_url_input():
    at = AppTest.from_file('main.py', default_timeout=30)
    at.run()
    at.sidebar.text_input[0].input('https://www.youtube.com/watch?v=wDmPgXhlDIg').run()
    at.sidebar.button[0].click().run()
    assert at.success[0].value == 'URL'


def test_bad_url_input():
    at = AppTest.from_file('main.py', default_timeout=30)
    at.run()
    at.sidebar.text_input[0].input('123').run()
    at.sidebar.button[0].click().run()
    assert len(at.sidebar.text_input) == 1
    assert at.warning[0].value == 'URL'
