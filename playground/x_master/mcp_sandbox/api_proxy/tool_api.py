import aiohttp
import asyncio
import os,sys,json
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
with open(os.path.join(current_dir, '../configs/mcp_config.json'), 'r') as f:
    mcp_config = json.load(f)

base_url = mcp_config['tool_api_url']


with open(os.path.join(current_dir, '../configs/web_agent.json'), 'r') as f:
    config = json.load(f)
    
async def web_search_api(session, query: str,top_k: int = 10):
    """Perform a web search via the search API.

    Args:
        session: aiohttp client session.
        query: Search query string.
        top_k: Maximum number of results to return.

    Returns:
        JSON response from the search API.
    """
    url = f"{base_url}/search"
    data = {
        "query": query,
        "serper_api_key": config['serper_api_key'],
        "top_k": top_k,
        "region": config['search_region'],
        "lang": config['search_lang'],
        "depth": 0
    }
    async with session.post(url, json=data) as resp:
        return await resp.json()


async def read_pdf_api(session, url: str):
    """Read PDF content from a URL via the read_pdf API.

    Args:
        session: aiohttp client session.
        url: URL of the PDF file to read.

    Returns:
        JSON response containing the PDF content.
    """
    server_url = f"{base_url}/read_pdf"
    data = {"url": url}
    async with session.post(server_url, json=data) as resp:
        return await resp.json()

async def fetch_web_api(session, url: str):
    """Fetch web page content from a URL via the fetch_web API.

    Args:
        session: aiohttp client session.
        url: URL of the web page to fetch.

    Returns:
        JSON response containing the web page content.
    """
    server_url = f"{base_url}/fetch_web"
    data = {"url": url}
    async with session.post(server_url, json=data) as resp:
        return await resp.json()


# Example main function
async def main():
    """Run example API calls for testing."""
    async with aiohttp.ClientSession() as session:
        # result = await web_search_api(session, "what is google")
        # print(result)

        # result = await read_pdf_api(session, "https://arxiv.org/pdf/2305.14342")
        # print(result)
        
        result = await fetch_web_api(session, "https://www.google.com/")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
