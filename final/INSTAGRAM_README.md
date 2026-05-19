# Instagram Profile Analyzer

This tool analyzes Instagram profiles to detect potential fake accounts using machine learning. It examines various profile attributes, engagement metrics, and behavioral patterns to assess the authenticity of an Instagram account.

## 🚀 Features

- **Profile Analysis**: Fetches and analyzes Instagram profile data
- **Engagement Metrics**: Calculates engagement rate based on recent posts
- **Fake Account Detection**: Uses machine learning to identify potential fake accounts
- **Comprehensive Report**: Provides detailed analysis with confidence scores
- **Interactive CLI**: Easy-to-use command-line interface

## 📋 Prerequisites

- Python 3.8+
- Instagram account (optional, for accessing private profiles)
- Internet connection

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Fake-Profile-Detection-using-ML
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r instagram_requirements.txt
   ```

4. (Optional) Create a `.env` file for Instagram credentials:
   ```
   INSTAGRAM_USERNAME=your_instagram_username
   INSTAGRAM_PASSWORD=your_instagram_password
   ```
   Note: Using credentials is optional but recommended for better functionality.

## 🚦 Usage

### Basic Usage

Run the analyzer with:
```bash
python instagram_analyzer.py
```

Then enter an Instagram username when prompted (with or without @ symbol).

### Command Line Arguments

You can also provide the username directly:
```bash
python instagram_analyzer.py username
```

### Example
```bash
python instagram_analyzer.py zuck
```

## 📊 What's Analyzed

The tool examines various aspects of an Instagram profile, including:

- **Profile Information**: Username, bio, profile picture
- **Activity**: Number of posts, followers, following
- **Engagement**: Likes, comments, and overall engagement rate
- **Account Age**: How long the account has been active
- **Behavioral Patterns**: Following/follower ratio, post frequency

## 🔍 How It Works

1. Fetches public profile data using web scraping
2. Analyzes recent posts for engagement patterns
3. Uses machine learning to assess the likelihood of the account being fake
4. Provides a detailed report with confidence scores

## ⚠️ Limitations

- Rate limiting by Instagram may affect data collection
- Private profiles require login credentials
- The accuracy depends on the available public data
- Instagram may block requests if too many are made in a short time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Instaloader](https://instaloader.github.io/) for Instagram data collection
- All contributors who helped improve this project
