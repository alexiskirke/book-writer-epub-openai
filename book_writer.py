import openai
from typing import List, Dict
import os
import json
import tiktoken
import re
from ebooklib import epub
import html
from abc import ABC, abstractmethod
from gpt4all import GPT4All
import sys
from PIL import Image, ImageDraw, ImageFont
import io
import base64

model_locations = "/Users/alexiskirke/Library/Application Support/nomic.ai/GPT4All/"
models = {"nous":"nous-hermes-llama2-13b.Q4_0.gguf", "meta-128k":"Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf","meta-instruct":"Meta-Llama-3-8B-Instruct.Q4_0.gguf"}

class LLMClient(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class GPT4AllClient(LLMClient):
    def __init__(self, model_name: str = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"):
        model_path = os.path.join(model_locations, model_name)
        self.model = GPT4All(model_path, device='gpu', n_ctx=64000)

    def generate_text(self, prompt: str, max_tokens: int = 900) -> str:
        response = self.model.generate(prompt, max_tokens=max_tokens)
        return response

class OutlineHTMLGenerator:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.outline_data = self._load_json()

    def _load_json(self) -> Dict:
        with open(self.json_file_path, 'r') as file:
            return json.load(file)

    def generate_html(self) -> str:
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Book Outline</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                h3 { color: #7f8c8d; }
                h4 { color: #95a5a6; }
                .paragraph { margin-left: 20px; }
            </style>
        </head>
        <body>
            <h1>Book Outline</h1>
        """

        html_content += self._generate_section(self.outline_data, 1)

        html_content += """
        </body>
        </html>
        """

        return html_content
    
    def _generate_section(self, section: Dict, depth: int) -> str:
        content = ""
        for title, subsections in section.items():
            cleaned_title = title.lstrip('-').strip('"')
            content += f"<h{depth}>{html.escape(cleaned_title)}</h{depth}>\n"
            if isinstance(subsections, dict):
                if "paragraphs" in subsections:
                    for paragraph in subsections["paragraphs"]:
                        content += f'<p class="paragraph">{html.escape(paragraph)}</p>\n'
                else:
                    content += self._generate_section(subsections, depth + 1)
            elif isinstance(subsections, str):
                content += f'<p class="paragraph">{html.escape(subsections)}</p>\n'
        return content

    def save_html(self, output_file_path: str):
        html_content = self.generate_html()
        with open(output_file_path, 'w') as file:
            file.write(html_content)
        print(f"HTML outline saved to {output_file_path}")

class OutlineEbookGenerator:
    def __init__(self, json_file_path: str, book_title: str = 'Book Outline', book_author: str = 'Outline Generator', generate_cover: bool = False, force_kindle_compatibility: bool = False, override_image_prompt: str = ''):
        self.json_file_path = json_file_path
        self.outline_data = self._load_json()
        self.book_title = book_title
        self.book_author = book_author
        self.generate_cover = generate_cover
        self.force_kindle_compatibility = force_kindle_compatibility
        self.override_image_prompt = override_image_prompt
        if self.generate_cover:
            self.openai_client = OpenAIClient()

    def _load_json(self) -> Dict:
        with open(self.json_file_path, 'r') as file:
            return json.load(file)

    def generate_ebook(self) -> epub.EpubBook:
        book = epub.EpubBook()
        book.set_identifier('book_outline_' + str(hash(json.dumps(self.outline_data))))
        book.set_title(self.book_title)
        book.set_language('en')
        book.add_author(self.book_author)

        spine = ['nav']
        toc = []

        # Create cover page if requested
        if self.generate_cover:
            cover_image = self._generate_cover_image()
            book.set_cover("cover.jpg", cover_image)

        # Create title page
        title_page = epub.EpubHtml(title='Title Page', file_name='title.xhtml')
        title_page.content = f'''
            <html>
            <head>
                <title>{html.escape(self.book_title)}</title>
            </head>
            <body>
                <h1 style="text-align: center;">{html.escape(self.book_title)}</h1>
                <p style="text-align: center;">{html.escape(self.book_author)}</p>
            </body>
            </html>
        '''
        book.add_item(title_page)
        spine.append(title_page)

        # Create table of contents
        toc_page = epub.EpubHtml(title='Table of Contents', file_name='toc.xhtml')
        toc_content = '<h1>Table of Contents</h1><nav epub:type="toc"><ol>'

        # Generate chapters
        for chapter_num, (chapter_title, chapter_content) in enumerate(self.outline_data.items(), 1):
            cleaned_chapter_title = chapter_title.strip().lstrip('-').replace('"', '')
            chapter = epub.EpubHtml(title=cleaned_chapter_title, file_name=f'chap_{chapter_num}.xhtml')
            chapter.content = f'<h1 id="{self._generate_id(cleaned_chapter_title)}">{html.escape(cleaned_chapter_title)}</h1>'
            chapter.content += self._generate_section(chapter_content, 2, chapter_num)
            
            book.add_item(chapter)
            spine.append(chapter)
            toc.append(chapter)

            # Add chapter to table of contents
            toc_content += f'<li><a href="chap_{chapter_num}.xhtml#{self._generate_id(cleaned_chapter_title)}">{html.escape(cleaned_chapter_title)}</a>'
            if isinstance(chapter_content, dict):
                toc_content += self._generate_toc_section(chapter_content, chapter_num, 2)
            toc_content += '</li>'

        toc_content += '</ol></nav>'
        toc_page.content = toc_content
        book.add_item(toc_page)
        spine.insert(2, toc_page)  # Insert TOC after cover and title page, before chapters

        book.spine = spine
        book.toc = toc
        book.add_item(epub.EpubNcx())
        if self.force_kindle_compatibility:
            book.add_item(epub.EpubNav())

        return book

    def _generate_section(self, section: Dict, depth: int, chapter_num: int) -> str:
        content = ""
        for title, subsections in section.items():
            cleaned_title = title.lstrip('-').replace('"', '')
            if depth <= 3:  # Only include h1, h2, and h3
                content += f'<h{depth} id="{self._generate_id(cleaned_title)}">{html.escape(cleaned_title)}</h{depth}>'
            if isinstance(subsections, dict):
                if "paragraphs" in subsections:
                    for paragraph in subsections["paragraphs"]:
                        content += self._process_paragraph(paragraph)
                else:
                    content += self._generate_section(subsections, depth + 1, chapter_num)
            elif isinstance(subsections, str):
                content += self._process_paragraph(subsections)
        return content

    def _generate_toc_section(self, section: Dict, chapter_num: int, depth: int) -> str:
        content = "<ol>"
        for title, subsections in section.items():
            cleaned_title = title.lstrip('-').replace('"', '')
            content += f'<li><a href="chap_{chapter_num}.xhtml#{self._generate_id(cleaned_title)}">{html.escape(cleaned_title)}</a>'
            if isinstance(subsections, dict) and "paragraphs" not in subsections and depth < 3:
                content += self._generate_toc_section(subsections, chapter_num, depth + 1)
            content += '</li>'
        content += "</ol>"
        return content

    def _generate_id(self, title: str) -> str:
        return re.sub(r'\W+', '-', title.lower())

    def _process_paragraph(self, paragraph: str) -> str:
        # Replace inline LaTeX equations with MathML
        paragraph = re.sub(r'\$(.+?)\$', lambda m: self._latex_to_mathml(m.group(1)), paragraph)
        return f'<p>{html.escape(paragraph)}</p>'

    def _latex_to_mathml(self, latex: str) -> str:
        # This is a placeholder function. You would need to implement or use a library
        # that converts LaTeX to MathML. For example, you could use the latex2mathml library.
        # Here's a simple example (you'd need to install latex2mathml first):
        from latex2mathml.converter import convert
        return convert(latex)
        
        # For now, we'll just wrap it in a math tag
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>{html.escape(latex)}</mtext></math>'
    
    def _generate_cover_image(self) -> bytes:
        if self.override_image_prompt:
            prompt = self.override_image_prompt
        else:
            prompt = f"Create an image inspired by the words: '{self.book_title}'. The image should have a clear point of focus. It should be photorealistic."
        try:
            response = self.openai_client.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1792",
                quality="standard",
                n=1,
            )
        except Exception as e:
            msg = f"Error: _generate_cover_image - OpenAI API call failed: {str(e)}"
            raise RuntimeError(msg) from e

        image_url = response.data[0].url
        
        # Use requests to download the image
        import requests
        image_response = requests.get(image_url)
        
        # Save the image
        image_path = f"{self.book_title}_cover_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_response.content)
        
        with open(image_path, "rb") as f:
            image_data = f.read()

            # Open the image using PIL
            img = Image.open(io.BytesIO(image_data))
        
        # Create a drawing object
        draw = ImageDraw.Draw(img)

        # chop the image so it is ratio 1.5:1
        width, height = img.size
        target_ratio = 1/1.5
        current_ratio = width / height

        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            img = img.crop((left, 0, left + new_width, height))
        elif current_ratio < target_ratio:
            # Image is too tall, crop height
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            img = img.crop((0, top, width, top + new_height))

        # Convert the image to RGB mode
        img = img.convert('RGB')
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

    def save_ebook(self, output_file_path: str):
        book = self.generate_ebook()
        epub.write_epub(output_file_path, book, {})
        print(f"Ebook-compatible outline saved to {output_file_path}")


class BookOutlineGenerator:
    def __init__(self, working_dir: str = 'outline_steps', use_gpt4all: bool = False):
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.full_outline = {}
        self.cumulative_tokens_sent = 0
        self.cumulative_tokens_received = 0
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.total_sections = 0
        self.completed_sections = 0
        self.use_gpt4all = use_gpt4all
        
        if use_gpt4all:
            self.llm_client = GPT4AllClient()
        else:
            self.llm_client = OpenAIClient()

    def send_prompt(self, topic: str, N0: int = 3, N1: int = 3, N2: int = 3, N3: int = 1, paragraph_style: str = '') -> Dict:
        print("Starting outline generation...")
        self.full_outline = {topic: {}}
        self._generate_chapters(topic, N0)
        self._generate_level1(N1)
        self._generate_level2(N2)
        self._generate_level3(N3, paragraph_style)
        print("Outline generation complete.")
        print(f"Cumulative tokens sent: {self.cumulative_tokens_sent}")
        print(f"Cumulative tokens received: {self.cumulative_tokens_received}")
        return self.full_outline

    def _generate_chapters(self, topic: str, num_chapters: int):
        print(f"Generating {num_chapters} chapters for topic: {topic}")
        prompt = f"Generate {num_chapters} chapter titles for a book about {topic}. Insure each title begins with the word Chapter and the chapter number, e.g. 'Chapter 1: ...'"
        response = self._send_request(prompt, is_title=True)
        chapters = response.strip().split('\n')
        self.full_outline[topic] = {chapter.strip(): {} for chapter in chapters[:num_chapters]}
        
        # Save chapters to file
        with open(os.path.join(self.working_dir, 'outline_depth_0.json'), 'w') as f:
            json.dump(self.full_outline, f, indent=2)
        
        print("Chapter generation complete.")

    def _generate_level1(self, N1: int):
        print(f"Generating level 1 sections")
        for chapter_title in self.full_outline[list(self.full_outline.keys())[0]]:
            context = self._get_outline_context()
            prompt = f"Given the following outline context:\n\n{context}\n\nGenerate {N1} section titles for the chapter titled: {chapter_title}, in such as a way as to avoid repeating material in earlier chapters or sections. Do not include the word 'Section' or a section number in the section titles."
            response = self._send_request(prompt, is_title=True)
            sections = response.strip().split('\n')
            self.full_outline[list(self.full_outline.keys())[0]][chapter_title] = {section.strip(): {} for section in sections[:N1]}
        
        # Save level 1 to file
        with open(os.path.join(self.working_dir, 'outline_depth_1.json'), 'w') as f:
            json.dump(self.full_outline, f, indent=2)
        
        print("Level 1 section generation complete.")

    def _generate_level2(self, N2: int):
        print(f"Generating level 2 sections")
        for chapter_title, chapter_content in self.full_outline[list(self.full_outline.keys())[0]].items():
            for section_title in chapter_content:
                context = self._get_outline_context()
                prompt = f"Given the following outline context:\n\n{context}\n\nGenerate {N2} subsection titles for the section titled: {section_title}, in such as a way as to avoid repeating material in earlier chapters or sections. Do not include the word 'subsection' or a subsection number in the subsection titles."
                response = self._send_request(prompt, is_title=True)
                subsections = response.strip().split('\n')
                self.full_outline[list(self.full_outline.keys())[0]][chapter_title][section_title] = {subsection.strip(): {} for subsection in subsections[:N2]}
        
        # Save level 2 to file
        with open(os.path.join(self.working_dir, 'outline_depth_2.json'), 'w') as f:
            json.dump(self.full_outline, f, indent=2)
        
        print("Level 2 section generation complete.")

    def _generate_level3(self, N3: int, paragraph_style: str):
        print(f"Generating level 3 sections with paragraphs")
        self._count_total_sections(self.full_outline)  # Count total sections before starting
        for chapter_title, chapter_content in self.full_outline[list(self.full_outline.keys())[0]].items():
            for section_title, section_content in chapter_content.items():
                for subsection_title in section_content:
                    context = self._get_outline_context()
                    prompt = f"You are a professional author. Given the following outline context:\n\n{context}\n\nWrite {N3} paragraphs about the topic: {subsection_title} (which you are an expert on), in such as a way as to avoid repeating concepts covered in earlier chapters or sections or paragraphs." + paragraph_style
                    response = self._send_request(prompt, is_title=False)
                    paragraphs = response.strip().split('\n\n')
                    self.full_outline[list(self.full_outline.keys())[0]][chapter_title][section_title][subsection_title] = {'paragraphs': paragraphs[:N3]}
                    
                    # Update progress
                    self.completed_sections += 1
                    self._print_progress()
        
        # Save level 3 to file
        with open(os.path.join(self.working_dir, 'outline_depth_3.json'), 'w') as f:
            json.dump(self.full_outline, f, indent=2)
        
        print("Level 3 section generation with paragraphs complete.")

    def _send_request(self, prompt: str, is_title: bool):
        tokens_sent = len(self.encoder.encode(prompt))
        self.cumulative_tokens_sent += tokens_sent
        prompt += "\nOnly respond with the content requested, no introduction or commentary."
        
        if self.use_gpt4all and is_title:
            response = self.llm_client.generate_text(prompt, max_tokens=200)
        else:
            response = self.llm_client.generate_text(prompt)
        
        tokens_received = len(self.encoder.encode(response))
        self.cumulative_tokens_received += tokens_received
        
        print(f"Cumulative tokens sent: {self.cumulative_tokens_sent}, Cumulative tokens received: {self.cumulative_tokens_received}")
        return response

    def _get_outline_context(self) -> str:
        return json.dumps(self.full_outline, indent=2)

    def _print_progress(self):
        percentage = (self.completed_sections / self.total_sections) * 100
        print(f"Progress: {percentage:.2f}% complete ({self.completed_sections}/{self.total_sections} sections)")

    def _count_total_sections(self, outline: Dict):
        self.total_sections = 0  # Reset the count
        for _, content in outline.items():
            if isinstance(content, dict):
                self._count_total_sections_recursive(content)

    def _count_total_sections_recursive(self, content: Dict):
        for _, subcontent in content.items():
            self.total_sections += 1
            if isinstance(subcontent, dict) and 'paragraphs' not in subcontent:
                self._count_total_sections_recursive(subcontent)

# Usage example:
title = 'How to become a Film Producer in the age of Indies, Netflix, YouTube and A.I.: A Step by Step Guide'

#generator = BookOutlineGenerator(use_gpt4all=False)  # Set to False to use OpenAI
paragraph_style = """Focus the writing 50 percent on actions that can be taken, and 50 percent on reflections on the deeper meaning of the topic. Write in a short snappy and modern style, but not too conversational. Write in shorter sentences, and avoid the passive voice, and adjectives and adverbs. Sound like a human, not a Large Language Model. Include examples occasionally. Include stories 
occasionally. If they are not true, say that they are illustrative. But
you can include true stories as well.
print("Starting book outline generation...")

outline = generator.send_prompt(title, N0=10, N1=4, N2=3, N3=3, 
                                paragraph_style=paragraph_style)
print("Book outline generation complete.")
print(outline)
"""

# Generate Ebook outline
filename = 'outline_depth_3.json'
filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outline_steps', filename)
image_prompt = "Photo realistic. A female middle-aged producer on a movie set watching from behind the camera while a film crew is shooting a scene."
ebook_generator = OutlineEbookGenerator(filename, book_title=title, book_author='Alexis Kirke', generate_cover=True, force_kindle_compatibility=True, override_image_prompt=image_prompt)
filename = title[:30] + '.epub'
ebook_generator.save_ebook(filename)