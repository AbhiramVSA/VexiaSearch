import asyncio, time, os, glob, json, httpx
from pathlib import Path

API = os.getenv('API_URL', 'http://localhost:8000')
USER_ID = os.getenv('BENCH_USER','bench-user')
PDF_DIR = os.getenv('BENCH_PDFS','./pdfs')

async def ingest():
    t0 = time.time()
    files = []
    for p in Path(PDF_DIR).glob('*.pdf'):
        files.append(('files', (p.name, open(p,'rb'), 'application/pdf')))
    async with httpx.AsyncClient(timeout=None) as client:
        data = {'user_id': USER_ID}
        r = await client.post(f"{API}/deploy", data=data, files=files)
        print('deploy', r.status_code, r.text)
    print('ingest request time', time.time()-t0)

async def chat(query: str):
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        r = await client.post(f"{API}/chat", json={'message': query, 'user_id': USER_ID})
        dt = (time.time()-t0)*1000
        print('chat', dt,'ms', r.json().get('answer')[:120])

async def main():
    await ingest()
    await asyncio.sleep(5)
    await chat('What is the main topic?')
    await chat('List key concepts.')

if __name__ == '__main__':
    asyncio.run(main())
