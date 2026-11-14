# Practice Sheet Studio

**Clean practice sheets from your book PDFs, without the solutions. In seconds.**

Upload a PDF.
The questions are picked out automatically.
The solutions are left behind.
You get a tidy practice sheet, ready to print.

---

## Why this app?

Preparing for exams takes a lot of time on one small chore:

> Finding the questions, and keeping your eyes away from the answers.

In most books, the solution sits right under the question.
Once you have seen it, you stop trying on your own.

This app removes that problem.

- Cuts out each question for you
- Leaves out the solutions
- Builds a neat, ready-to-use PDF

No scissors. No cut and paste. No fuss.

---

## Who is it for?

- **Students:** make your own practice sheets and test yourself without peeking at answers
- **Teachers and coaching centres:** pick questions from a book and print a class test in minutes
- **Study groups:** make one sheet and solve it together
- **Anyone who likes control:** draw your own boxes and choose exactly which questions go in

---

## Features

### Smart scan

- Reads each page and finds the questions
- Understands **Bangla and English** text
- Spots question starts from numbers and markers (Q, Ex, Problem, প্রশ্ন, ❑ and more)
- Skips solutions when it sees Sol, Ans, Answer, Hint, উত্তর, সমাধান, ব্যাখ্যা and similar words

### Three page modes

- **Auto:** the app decides for you
- **Book:** regular book-style pages
- **Slide:** questions inside boxes, as on slides

### One or two columns

- Arrange the output in one column or two
- Handles slanted divider lines between columns

### Box editor

If the scan misses something, fix it by hand:

- Add a new box
- Drag the edges to resize a box
- Delete a wrong box
- Select several boxes at once
- Move between pages
- Select every box on a page in one tap

### Batch edit

When many boxes need the same fix, do it once:

- Choose all boxes, left column, right column, or one page
- Shift boxes left, right, up, or down
- Make them wider, narrower, taller, or shorter

### Output settings

- Set your own **sheet title**
- Adjust the **gap** between questions
- Turn on **white background** to make dark pages printer-friendly
- Scan only the pages you want, for example `1-5`

### PC and phone modes

- On PC, the editor opens on the right side
- On phone, box edges are easier to grab with a finger
- Separate zoom and pan buttons are included

### A calm workspace

- Slow, soft fog in the background
- Very gentle, quiet sounds
- Use the speaker button at the bottom left to change the volume or turn sounds off

---

## How to use it

1. **Upload your PDF**
   - Drag the file in, or press Upload

2. **Check the settings on the left**
   - Change them if you need to
   - The defaults work fine too

3. **Press "স্ক্যান করুন" (Scan)**
   - Wait a moment
   - A progress card shows how far it has got

4. **Review the result**
   - Open the "বক্স এডিটর" (Box Editor) if anything needs fixing

5. **Press "ডাউনলোড (PDF)" (Download)**
   - Your practice sheet is ready

---

## Tips for better results

- Use a clear, sharp PDF
- Try one or two pages first, then the whole book
- Pick the page mode that matches your book
- Glance over the result before printing
- If your book uses different markers, add them to the tag files (see below)

---

## Adding your own tags

The words that mark questions and solutions live in two plain text files.

| File | What it holds |
|------|---------------|
| `question_tags.txt` | Words that start a question (Q, Ex, প্রশ্ন, ...) |
| `solution_tags.txt` | Words that start a solution or a part to skip (Sol, উত্তর, ...) |

Write one word per line and save the file.
No code changes needed.

---

## Run it on your own computer

```bash
# 1. Install the Python packages
pip install -r requirements.txt

# 2. Install Bangla OCR (Linux)
sudo apt install tesseract-ocr tesseract-ocr-ben

# 3. Start the app
streamlit run app.py
```

> On Streamlit Cloud, the `packages.txt` file takes care of step 2 for you.

---

## What each file does

| File | Purpose |
|------|---------|
| `app.py` | The main app |
| `canvas_editor.html` | The box editor |
| `style.css` | Look and feel |
| `ambient.css` | Fog background styling |
| `sfx.js` | Soft sounds and fog movement |
| `question_tags.txt` | Question markers |
| `solution_tags.txt` | Solution markers |
| `requirements.txt` | Python packages |
| `packages.txt` | System packages (OCR) |

---

## One honest note

The app is accurate, but not perfect.
On blurry scans or unusual print, a question or two may be missed.

So take a quick look before you print.
The box editor is there to fix anything in seconds.

---

**Study calmly, study neatly, study your way.**


---
## ভিশন ইঞ্জিন (নতুন, ফটো/স্ক্যান পেজের জন্য)
- ফাইল: `question_vision.py` (ইঞ্জিন), `vision_bridge.py` (app.py-র সাথে সংযোগ), `vision_profile.json` (সেটিং)।
- সাইডবার → পেজ সেটআপ → **ইঞ্জিন**: "ভিশন" (ডিফল্ট) বা "পুরনো"। ভিশন কিছু না পেলে নিজে থেকেই পুরনো ইঞ্জিনে চলে।
- কীভাবে কাজ করে: পেজ সোজা করে (ছবির বাঁকা/কোনাকুনি ঠিক করে), প্রতিটি কলামের বাম margin ধরে নামে। `উত্তর:` / `ব্যাখ্যা:` / `সমাধান:` লেবেল বইয়ের ভেতরেই বারবার আসে — ইঞ্জিন নিজেই সেগুলো চিনে নেয়। লেবেলের পরের প্রথম ছোট margin-লাইন = নতুন প্রশ্ন। প্রশ্ন = নম্বর-লাইন থেকে লেবেলের ঠিক আগ পর্যন্ত।
- বাংলা OCR লাগে না।
- বক্স এডিটরে বক্স না সরালে প্র্যাকটিস শিটে সোজা-করা পরিষ্কার ছবি যায়। বক্স সরালে সেটা সাধারণ ক্রপে ফিরে যায়।
- নতুন বইয়ে সমস্যা হলে আগে `vision_profile.json`-এর মান বদলে দেখুন (`label_thr`, `label_frac`, `margin_tol`, `num_max_w`)।
- এখন পর্যন্ত শুধু Rhombus (ACS) বইয়ের ফিজিক্স ও কেমিস্ট্রির পেজে পরীক্ষা করা হয়েছে।
