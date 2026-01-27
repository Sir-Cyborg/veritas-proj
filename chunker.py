from pathlib import Path
import argparse


class Chunker:
    """
    Chunking by numbered headings (e.g., '1. Introduction', '5.3 Secure Coding').
    Strategy:
      - detect heading lines
      - group text until the next heading
      - write each group as a chunk file
    """

    def __init__(self):
        pass

    def is_heading(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return False

        first_token = line.split()[0]  # e.g. "5.3" or "1."
        first_token = first_token.rstrip(".")  # "1." -> "1"
        parts = first_token.split(".")         # "5.3" -> ["5","3"]

        # heading if all parts are digits (e.g., 1, 2, 5.3, 10.2.1)
        return all(p.isdigit() for p in parts)

    def split_into_chunks(self, text: str) -> list[tuple[str, str]]:
        """
        Returns a list of (chunk_title, chunk_text).
        chunk_title is the heading line, used for naming/debug.
        """
        lines = text.splitlines()

        chunks: list[tuple[str, str]] = []
        current_title = "0_FULL_DOCUMENT"
        current_lines: list[str] = []

        for line in lines:
            if self.is_heading(line):
                # save previous chunk
                if current_lines:
                    chunks.append((current_title, "\n".join(current_lines).strip()))
                # start new chunk
                current_title = line.strip()
                current_lines = [line.strip()]  # include heading line in chunk
            else:
                current_lines.append(line)

        # save last chunk
        if current_lines:
            chunks.append((current_title, "\n".join(current_lines).strip()))

        return chunks

    def chunk_folder(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(input_dir.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in: {input_dir}")
            return

        for file_path in txt_files:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = self.split_into_chunks(text)

            base = file_path.stem
            for i, (title, chunk_text) in enumerate(chunks, start=1):
                safe_title = self._safe_name(title)
                out_name = f"{base}__chunk_{i:03d}__{safe_title}.txt"
                (output_dir / out_name).write_text(chunk_text + "\n", encoding="utf-8")

            print(f"- {file_path.name}: {len(chunks)} chunks written")

    def _safe_name(self, s: str, max_len: int = 40) -> str:
        # very simple filename sanitization
        s = s.lower()
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
        s = s.replace(" ", "_")
        s = "".join(ch if ch in allowed else "_" for ch in s)
        return s[:max_len] if len(s) > max_len else s




def main() -> None:
    parser = argparse.ArgumentParser(description="Simple chunking for .txt documents.")
    parser.add_argument("--input", required=True, help="Folder with .txt documents")
    parser.add_argument("--output", default="chunks", help="Output folder (default: chunks)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    chunker = Chunker()
    chunker.chunk_folder(input_dir, output_dir)


if __name__ == "__main__":
    main()
