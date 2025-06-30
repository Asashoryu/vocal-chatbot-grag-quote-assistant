import xml.etree.ElementTree as ET


def print_structure(element, indent=0):
    print("  " * indent + f"<{element.tag}>")
    for child in element:
        print_structure(child, indent + 1)


def main():
    file_path = "/home/olexandro/NLP/vocal-chatbot-grag-quote-assistant/storage/xml/wikimedia_quotes.xml"

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print(f"Root: <{root.tag}>")
        print_structure(root)
    except ET.ParseError as pe:
        print(f"XML Parse Error: {pe}")
    except FileNotFoundError:
        print("The file was not found. Please check the path.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
