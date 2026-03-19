#!/usr/bin/env python3
"""
MCP Adapter - Converts your search API to MCP protocol.
Calls your existing FastAPI service.
"""

import os
import json
import asyncio
from mcp.server.fastmcp import FastMCP
import aiohttp

# Configuration
MCP_PORT = int(os.getenv("MCP_PORT", "8002"))
HOST = os.getenv("HOST", "0.0.0.0")

# Your FastAPI service address
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:1234")

# Read API key from config file or environment variable
def _load_serper_api_key():
    """Read Serper API key from environment variable or config file."""
    key = os.getenv("SERPER_API_KEY")
    if key:
        return key
    # Try to read from config file
    config_path = os.path.join(os.path.dirname(__file__), "../configs/web_agent.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get("serper_api_key", "")
    return ""

SERPER_API_KEY = _load_serper_api_key()

# Initialize MCP server
mcp = FastMCP(
    "search-tools",
    host=HOST,
    port=MCP_PORT,
)

async def make_async_request(session, url, payload, timeout=30):
    """Async HTTP request."""
    async with session.post(url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        return await response.json()

@mcp.tool()
async def search(
    query: str,
    top_k: int = 10,
    region: str = "us",
    lang: str = "en",
    depth: int = 0
) -> str:
    """
    Search web content.

    Args:
        query: Search keywords
        top_k: Number of results to return, default 10
        region: Search region (us, uk, cn, etc.)
        lang: Language (en, zh-CN, etc.)
        depth: Search depth (brief, basic, detailed)
    """
    try:
        payload = {
            "query": query,
            "serper_api_key": SERPER_API_KEY,
            "top_k": top_k,
            "region": region,
            "lang": lang,
            "depth": depth
        }
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session, 
                f"{API_BASE_URL}/search", 
                payload
            )
        
        # Format output
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP请求错误: {str(e)}"
    except asyncio.TimeoutError:
        return "请求超时，请稍后重试"
    except Exception as e:
        return f"搜索出错: {str(e)}"

@mcp.tool()
async def read_pdf(url: str) -> str:
    """
    Read PDF file content from a URL.

    Args:
        url: URL of the PDF file
    """
    try:
        payload = {"url": url}
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session,
                f"{API_BASE_URL}/read_pdf",
                payload,
                timeout=60
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP请求错误: {str(e)}"
    except asyncio.TimeoutError:
        return "请求超时，PDF文件可能较大"
    except Exception as e:
        return f"读取PDF出错: {str(e)}"

@mcp.tool()
async def fetch_web(url: str) -> str:
    """
    Fetch web page content.

    Args:
        url: URL of the web page
    """
    try:
        payload = {"url": url}
        
        async with aiohttp.ClientSession() as session:
            result = await make_async_request(
                session,
                f"{API_BASE_URL}/fetch_web",
                payload
            )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except aiohttp.ClientError as e:
        return f"HTTP请求错误: {str(e)}"
    except asyncio.TimeoutError:
        return "请求超时，请检查URL是否有效"
    except Exception as e:
        return f"获取网页内容出错: {str(e)}"

if __name__ == "__main__":

    # Run MCP server
    mcp.run(transport="streamable-http") 
