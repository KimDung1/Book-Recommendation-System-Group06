import zipfile
import xml.etree.ElementTree as ET

def extract(path):
    with zipfile.ZipFile(path) as docx:
        tree = ET.fromstring(docx.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with open('output.txt', 'w', encoding='utf-8') as f:
        for p in tree.findall('.//w:p', ns):
            text = ''.join([t.text for r in p.findall('.//w:r', ns) if (t := r.find('w:t', ns)) is not None and t.text])
            if text: f.write(text + '\n')

extract(r'e:\BigData\Group 06\Instructions_Group_06.docx')
