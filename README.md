# Valorant Aim Coach

A single-file, Gen-AI–powered Valorant aim coaching web app. Answer a few questions
about your setup and struggles and get a personalized, AI-generated sensitivity guide
and daily aim-training plan — plus a live chat with your AI coach.

Powered by **Google Gemini** (free tier). No build step, no dependencies — it's one
self-contained `.html` file.

## Features

- 🤖 **AI coaching plan** — streamed, personalized advice based on your profile (DPI,
  sensitivity, eDPI, rank, experience, and self-reported aim problems).
- 💬 **Live chat** — ask your AI coach follow-up questions with full conversation context.
- 🎮 **Built-in tools** — sensitivity converter, crosshair builder, target trainer, and
  a sensitivity calibration game.
- 🔌 **Offline fallback** — works without a key using built-in templates.

## Usage

1. Open `valorant-aim-coach_3.html` in any modern browser.
2. Click **⚡ Connect AI** (top right).
3. Paste a free Google Gemini API key from
   [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
   (starts with `AIza...`).
4. Run the coach flow.

## Security note

This is a client-side app. Any API key is stored only in your browser's `localStorage`
and sent directly to Google from your device. **Do not commit a real API key to a public
repository** — keep it out of the source, or use the in-app "Connect AI" button instead.

## License

Not affiliated with Riot Games. For personal/educational use.
