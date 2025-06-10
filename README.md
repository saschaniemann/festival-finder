# Festival Finder

Festival Finder is a web application designed to help users discover music festivals around the world. Whether you're a fan of rock, pop, electronic, or indie music, this app makes it easy to find festivals that match your preferences.

![Festival Finder Demo](assets/demo.gif)

## Features

- **Search Festivals**: Find festivals by location, date, or genre.
- **Sort Festivals**: Sort the found festivals by name, date or distance
- **Festivals on map**: Show the festivals on a map
- **Festival Details**: View detailed information about each festival, including lineup, venue, and links to its official website.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/saschaniemann/festival-finder.git
    ```
2. Navigate to the project directory:
    ```bash
    cd festival-finder
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the crawler. This might take a couple of hours (mostly limited by Gemini's and Geoapify's API limitations of the free tier):
    ```bash
    python crawler.py
    ```
2. Start the application:
    ```bash
    streamlit run app.py
    ```
3. Open the application in your browser (usually at `http://localhost:8501`).
4. Use the filters to search for festivals by location, genre, date, or bands.
5. View festival details or explore the map for nearby events.

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python
- **APIs**: Geoapify (geocoding), Gemini API (lineup extraction)

## Data Sources

- [Festivalticker](https://www.festivalticker.de): Festival information
- The festivals' official web pages as linked at Festivalticker
- [Geoapify](https://www.geoapify.com): Geocoding API
- [Gemini API](https://developers.google.com/): Lineup extraction

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, feel free to reach out.

Enjoy discovering your next festival adventure with Festival Finder!