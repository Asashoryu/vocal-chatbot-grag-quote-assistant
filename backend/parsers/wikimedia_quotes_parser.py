#!/usr/bin/env python

import os
import json
import re
import xmltodict
from io import StringIO
from html.parser import HTMLParser
from unidecode import unidecode
from langdetect import detect
import unwiki

# === Configuration ===
input_file = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/xml/wikimedia_quotes.xml"
MAX_LINE_LEN = 100000
cutoffArg = MAX_LINE_LEN
langArg = "en"
output_json_file = "quotes_output.jsonl"

# === Regex pattern for section headers ===
section_header_pattern = re.compile(r'^={3,}\s*.*?\s*={3,}$')


# === Utilities ===

def clean_text(text, remove_prefix_len=0):
    """Clean a line: unwiki, remove HTML, unicode, excess whitespace, and optional prefix."""
    cleaned = unwiki.loads(text)
    cleaned = strip_tags(cleaned)
    cleaned = unidecode(cleaned)
    cleaned = cleaned.replace("\\'", "").replace('"', '')
    cleaned = ' '.join(cleaned.split())
    if remove_prefix_len > 0:
        cleaned = cleaned[remove_prefix_len:]
    return cleaned


def truncate_source(text, max_len=MAX_LINE_LEN):
    """Truncate source gracefully without cutting off mid-word."""
    if len(text) > max_len:
        truncated = text[:max_len].rsplit(' ', 1)[0] + "..."
        return truncated
    return text


class MLStripper(HTMLParser):
    """HTML tag stripper."""

    def __init__(self):
        super().__init__()
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    """Strip HTML tags from text."""
    stripper = MLStripper()
    stripper.feed(html)
    return stripper.get_data()


# === Main Quote Extraction ===

def writeQuotes(content):
    quoteList = []
    write = False
    i = 0
    current_section_source = ""

    while i < len(content):
        line = content[i].strip()

        if line.startswith('==') and not line.startswith('==='):
            write = False  # End of quotes section if main section starts

        if line in ('==Quotes==', '== Quotes =='):
            write = True

        # Detect sub-section headers (context)
        if write and section_header_pattern.match(line):
            inner = re.sub(r'^={3,}\s*|\s*={3,}$', '', line).strip()
            section_source = clean_text(inner)
            section_source = truncate_source(section_source)
            current_section_source = section_source
            i += 1
            continue

        # Match any line starting with * or more, optionally followed by spaces
        match = re.match(r'^(\*+)\s*(.*)', line)
        if write and match:
            stars, quote_text = match.groups()
            level = len(stars)

            # If it's a top-level quote
            if level == 1:
                cleaned_line = clean_text(quote_text)
                source_lines = []
                j = i + 1

                while j < len(content):
                    sub_line = content[j].strip()
                    sub_match = re.match(r'^(\*{2,})\s*(.*)', sub_line)
                    if not sub_match:
                        break

                    _, sub_text = sub_match.groups()
                    cleaned_src = clean_text(sub_text)
                    cleaned_src = truncate_source(cleaned_src)
                    if cleaned_src:
                        source_lines.append(cleaned_src)
                    j += 1

                if "://" not in cleaned_line and len(cleaned_line) < cutoffArg:
                    try:
                        if langArg == "all" or detect(cleaned_line) == langArg:
                            quote_entry = {"quote": cleaned_line}
                            if current_section_source:
                                quote_entry["context"] = current_section_source
                            if source_lines:
                                quote_entry["sources"] = [
                                    {"text": s} for s in source_lines]
                            quoteList.append(quote_entry)
                    except:
                        pass

                i = j - 1  # Skip processed source lines

        i += 1

    return quoteList


# === XML Parsing Handler ===

def handle(_, value):
    try:
        content = str(value['revision']['text']).split('\\n')
        quotes = writeQuotes(content)
        if quotes:
            title = str(value['title'])
            with open(output_json_file, "a", encoding="utf-8") as f:
                for q in quotes:
                    entry = {
                        "author": title,
                        "quote": q["quote"]
                    }
                    if "context" in q:
                        entry["context"] = q["context"]
                    if "sources" in q:
                        entry["sources"] = q["sources"]
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return True


# === Run the parser ===

try:
    xmltodict.parse(open(input_file, "rb"), item_depth=2, item_callback=handle)
except KeyboardInterrupt:
    print("\nParsing interrupted by user (Ctrl+C). Exiting...")
    exit(0)
