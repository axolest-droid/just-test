# -*- coding: utf-8 -*-
"""
info_panel.py  —  সাইডবারের "ⓘ অ্যাপ সম্পর্কে" পপওভার
(নতুন ফাইল: app.py এর সাইডবারের শেষে ২-৩ লাইন দিয়ে এটাকে ডাকা হয়েছে)

লেখা বদলাতে চাইলে নিচের FEATURES, STEPS, TIPS তালিকা এডিট করুন।
সরাতে চাইলে app.py থেকে "info_panel" ব্লকটি মুছে দিন।
"""
import streamlit as st

# --- আইকন (অ্যাপের বাকি আইকনের মতোই চিকন লাইনের) -------------------------
def _svg(inner):
    return ('<svg width="15" height="15" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round">' + inner + '</svg>')

ICONS = {
    "scan":   _svg('<circle cx="11" cy="11" r="7"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>'),
    "skip":   _svg('<path d="M4 4l16 16"></path><circle cx="12" cy="12" r="9"></circle>'),
    "pages":  _svg('<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline>'),
    "edit":   _svg('<rect x="3" y="3" width="18" height="18" rx="2"></rect><path d="M8 12h8"></path><path d="M12 8v8"></path>'),
    "batch":  _svg('<line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line>'),
    "device": _svg('<rect x="5" y="2" width="14" height="20" rx="2"></rect><line x1="12" y1="18" x2="12.01" y2="18"></line>'),
    "sheet":  _svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline>'),
}

# --- লেখা (এখান থেকে বদলান) ---------------------------------------------------
FEATURES = [
    ("scan",   "স্মার্ট স্ক্যান",    "বাংলা ও ইংরেজি পিডিএফ থেকে প্রশ্ন নিজে চিনে নেয়।"),
    ("skip",   "সমাধান বাদ",        "Sol, Ans, উত্তর, সমাধান দেখলে সেই অংশ ছেড়ে দেয়।"),
    ("pages",  "পাতার ধরন",         "অটো, বই বা স্লাইড। এক বা দুই কলাম।"),
    ("edit",   "বক্স এডিটর",        "বক্স যোগ করুন, টেনে বদলান, ভুল হলে মুছুন।"),
    ("batch",  "ব্যাচ এডিট",        "সব বক্স একসাথে সরান বা মাপ বদলান।"),
    ("sheet",  "নিজের মতো শিট",     "শিরোনাম, ফাঁক ও সাদা ব্যাকগ্রাউন্ড বেছে নিন।"),
    ("device", "পিসি ও ফোন",        "দুই মোডেই আরামে চলে।"),
]

STEPS = [
    "পিডিএফ আপলোড করুন",
    "“স্ক্যান করুন” চাপুন",
    "দরকার হলে বক্স এডিটরে ঠিক করুন",
    "PDF ডাউনলোড করুন",
]

TIPS = [
    "বক্স বেছে <b>Delete</b> চাপলে মুছে যায়।",
    "<b>Shift</b> বা <b>Ctrl</b> চেপে অনেক বক্স বাছা যায়।",
    "নিচের বাম কোণের বাটনে শব্দ কমান বা বন্ধ করুন।",
]

NOTE = "ওসিআর ১০০% নিখুঁত নয়। ছাপার আগে একবার দেখে নিন।"

# --- সাজসজ্জা (style.css-এর রঙ ও ফন্ট ব্যবহার করে) --------------------------
CSS = """
.ps-info{width:min(330px,76vw);max-height:68vh;overflow-y:auto;padding:.1rem .15rem .1rem 0;color:var(--text-primary,#E6EDF3);}
.ps-info *{box-sizing:border-box;}
.ps-info-head{display:flex;align-items:center;justify-content:space-between;gap:.6rem;padding-bottom:.65rem;margin-bottom:.7rem;border-bottom:1px solid var(--border-subtle,rgba(255,255,255,.07));}
.ps-info-title{font-size:.95rem;font-weight:700;letter-spacing:-.01em;line-height:1.2;}
.ps-info-sub{font-size:.72rem;color:var(--text-muted,#8B949E);margin-top:.15rem;line-height:1.35;}
.ps-info-pill{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--accent-sage,#5E9E8B);background:var(--accent-tint,rgba(94,158,139,.12));border:1px solid var(--accent-border,rgba(94,158,139,.32));padding:.12rem .45rem;border-radius:5px;white-space:nowrap;}
.ps-info-sec{font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-sage,#5E9E8B);margin:.85rem 0 .45rem 0;}
.ps-info-row{display:flex;gap:.6rem;align-items:flex-start;padding:.38rem 0;}
.ps-info-ico{flex:0 0 auto;width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--accent-sage,#5E9E8B);background:var(--accent-tint,rgba(94,158,139,.12));border:1px solid var(--accent-border,rgba(94,158,139,.32));}
.ps-info-name{font-size:.8rem;font-weight:600;line-height:1.25;}
.ps-info-desc{font-size:.72rem;color:var(--text-muted,#8B949E);line-height:1.4;margin-top:.1rem;}
.ps-info-steps{background:var(--bg-carbon,#0B0E14);border:1px solid var(--border-subtle,rgba(255,255,255,.07));border-radius:8px;padding:.45rem .7rem;}
.ps-info-step{display:flex;gap:.6rem;align-items:baseline;font-size:.76rem;padding:.2rem 0;}
.ps-info-num{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--accent-sage,#5E9E8B);}
.ps-info-tip{display:flex;gap:.5rem;font-size:.72rem;color:var(--text-muted,#8B949E);line-height:1.45;padding:.14rem 0;}
.ps-info-tip:before{content:"";flex:0 0 auto;width:4px;height:4px;border-radius:50%;background:var(--accent-sage,#5E9E8B);margin-top:.5rem;}
.ps-info-tip b{color:var(--text-primary,#E6EDF3);font-weight:600;}
.ps-info-note{margin-top:.8rem;padding-top:.6rem;border-top:1px solid var(--border-subtle,rgba(255,255,255,.07));font-size:.7rem;color:var(--text-muted,#8B949E);}
"""


def build_info_html():
    # নোট: st.markdown-এ ফাঁকা লাইন বা ৪-স্পেস ইনডেন্ট থাকলে HTML ভেঙে যায়,
    # তাই সবকিছু এক লাইনে জোড়া হয়েছে।
    h = ['<style>', CSS, '</style>', '<div class="ps-info">']
    h.append(
        '<div class="ps-info-head"><div>'
        '<div class="ps-info-title">Practice Sheet Studio</div>'
        '<div class="ps-info-sub">সমাধান বাদ দিয়ে পরিচ্ছন্ন প্র্যাকটিস শিট</div>'
        '</div><span class="ps-info-pill">ABOUT</span></div>'
    )
    h.append('<div class="ps-info-sec">মূল সুবিধা</div>')
    for key, name, desc in FEATURES:
        h.append(
            '<div class="ps-info-row"><div class="ps-info-ico">' + ICONS[key] + '</div>'
            '<div><div class="ps-info-name">' + name + '</div>'
            '<div class="ps-info-desc">' + desc + '</div></div></div>'
        )
    h.append('<div class="ps-info-sec">চার ধাপে</div><div class="ps-info-steps">')
    for i, s in enumerate(STEPS, 1):
        h.append('<div class="ps-info-step"><span class="ps-info-num">0' + str(i) + '</span><span>' + s + '</span></div>')
    h.append('</div>')
    h.append('<div class="ps-info-sec">ছোট টিপস</div>')
    for t in TIPS:
        h.append('<div class="ps-info-tip"><span>' + t + '</span></div>')
    h.append('<div class="ps-info-note">' + NOTE + '</div></div>')
    return "".join(h)


def render_info_panel():
    with st.popover("ⓘ  অ্যাপ সম্পর্কে", use_container_width=True):
        st.markdown(build_info_html(), unsafe_allow_html=True)
