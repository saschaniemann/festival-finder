"""Update the default meta tags of Streamlit apps. This is needed to add og:title, og:image, and so on. Based on https://discuss.streamlit.io/t/adding-a-meta-description-to-your-streamlit-app/17847/16."""

import fileinput
import sys
import os

NEW_TITLE = """
    <title>Festival Finder</title>
    <meta name="title" content="Festival Finder">
    <meta name="description" content="Festival Finder is a Streamlit app that helps you find music festivals based on your preferences.">
    <meta name="keywords" content="festival, music, finder, app">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://saschaniemann.com/festival-finder/">
    <meta property="og:title" content="Festival Finder">
    <meta property="og:description" content="Festival Finder is a Streamlit app that helps you find music festivals based on your preferences.">
    <meta property="og:image" content="https://saschaniemann.com/festival-finder/app/static/og_image.png">
    <meta property="twitter:card" content="Festival Finder is a Streamlit app that helps you find music festivals based on your preferences.">
    <meta property="twitter:url" content="https://saschaniemann.com/festival-finder/">
    <meta property="twitter:title" content="Festival Finder">
    <meta property="twitter:description" content="Festival Finder is a Streamlit app that helps you find music festivals based on your preferences.">
    <meta property="twitter:image" content="https://saschaniemann.com/festival-finder/app/static/og_image.png">
"""

STREAMLIT_INDEX = "/usr/local/lib/python3.13/site-packages/streamlit/static/index.html"


def replace_title():
    """Replace the title tag in the Streamlit index.html file with a new title and meta tags."""
    try:
        # Create a backup of the original file
        backup_file = STREAMLIT_INDEX + ".bak"
        if not os.path.exists(backup_file):
            with open(STREAMLIT_INDEX, "r") as original:
                with open(backup_file, "w") as backup:
                    backup.write(original.read())

        # Replace the title tag in the file
        with fileinput.FileInput(STREAMLIT_INDEX, inplace=True, backup=".bak") as f:
            for line in f:
                print(line.replace("<title>Streamlit</title>", NEW_TITLE), end="")

        print(f"Successfully updated title in {STREAMLIT_INDEX}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    replace_title()
