# Autobot

Autobot is an AI-powered assistant designed to help Italian customers compare and choose Chinese cars available in Italy. By leveraging a structured database and advanced language models, Autobot provides clear, neutral, and practical answers about car models, features, prices, and comparisons.

## Features

- Ask simple or complex questions about Chinese cars in Italy
- Get details on versions, prices, features, and technical specs
- Compare different models side by side
- Data-driven answers based on a real CSV database
- Optimized for non-technical users

## Technologies

- Python
- pandas
- LangChain / OpenAI / Llama
- CSV data

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/pettinellasd/autobot.git
   cd autobot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenAI or Together API key to a `.env` file:
   ```
   TOGETHER_API_KEY=your_api_key_here
   ```

4. Run the assistant:
   ```bash
   python autobot_env/main.py
   ```

## Data

The car database is stored in `autobot_env/auto_dati.csv`.  
You can update or expand this file to include more models and features.

## Topics

`ai-assistant`, `chatbot`, `cars`, `chinese-cars`, `italy`, `automotive`, `recommendation-system`, `pandas`, `langchain`, `openai`, `consumer-tech`, `data-driven`

## License

MIT

---

Help Italian consumers make informed choices in the growing market of Chinese cars!