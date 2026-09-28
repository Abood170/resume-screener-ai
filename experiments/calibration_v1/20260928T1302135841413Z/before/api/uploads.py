"""Stream multipart data directly into bounded RAM, never UploadFile disk spools."""

from fastapi import Request
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from src.extract import MAX_FILE_BYTES, FileValidationError, validate_extension

MAX_BODY_BYTES = MAX_FILE_BYTES + 64 * 1024


async def read_cv_upload(request: Request) -> tuple[str, bytes]:
    media_type, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary")
    if media_type != b"multipart/form-data" or not boundary or len(boundary) > 200:
        raise FileValidationError(
            "Send a multipart/form-data request with one field named 'file'.", 400
        )
    buffer = bytearray()
    header_name = bytearray()
    header_value = bytearray()
    headers: dict[bytes, bytes] = {}
    filename: str | None = None
    parts = 0
    completed = False

    def part_begin():
        nonlocal parts
        parts += 1
        if parts > 1:
            raise FileValidationError(
                "Upload exactly one file using the 'file' field.", 400
            )

    def header_field(data, start, end):
        header_name.extend(data[start:end])

    def header_data(data, start, end):
        header_value.extend(data[start:end])

    def header_end():
        headers[bytes(header_name).lower()] = bytes(header_value)
        header_name.clear()
        header_value.clear()

    def headers_finished():
        nonlocal filename
        disposition, fields = parse_options_header(
            headers.get(b"content-disposition", b"")
        )
        if (
            disposition != b"form-data"
            or fields.get(b"name") != b"file"
            or not fields.get(b"filename")
        ):
            raise FileValidationError(
                "Upload exactly one file using the 'file' field.", 400
            )
        filename = fields[b"filename"].decode("utf-8", errors="replace")
        validate_extension(filename)

    def part_data(data, start, end):
        if len(buffer) + end - start > MAX_FILE_BYTES:
            raise FileValidationError(
                "File exceeds the 5 MB (5,242,880 bytes) limit.", 413
            )
        buffer.extend(data[start:end])

    def end():
        nonlocal completed
        completed = True

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": part_begin,
            "on_header_field": header_field,
            "on_header_value": header_data,
            "on_header_end": header_end,
            "on_headers_finished": headers_finished,
            "on_part_data": part_data,
            "on_end": end,
        },
        max_header_count=8,
        max_header_size=8192,
    )
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                raise FileValidationError(
                    "Upload exceeds the 5 MB file limit and multipart overhead allowance.",
                    413,
                )
            parser.write(chunk)
        parser.finalize()
    except MultipartParseError:
        raise FileValidationError("Malformed multipart upload.", 400) from None
    if not completed or filename is None:
        raise FileValidationError(
            "Incomplete multipart upload or missing 'file' field.", 400
        )
    return filename, bytes(buffer)
