#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
import requests
import ast
from duckduckgo_search import DDGS
# load_dotenv()


class GoogleSearchEngine():
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.service = build("customsearch", "v1", developerKey=self.api_key)
        
    def search(self, query: str, num: int = 3):
        try:
            res = self.service.cse().list(q=query, cx=self.cse_id, num=num).execute()
            return '\n'.join([item['snippet'] for item in res['items']])
        except:
            return ''


class SearchAPIEngine():

    def search(self, query: str, item_num: int = 3):
            try:
                url = "https://www.searchapi.io/api/v1/search"
                params = {
                "engine": "google",
                "q": query,
                "api_key": os.getenv("SEARCHAPI_API_KEY")
                }

                response = ast.literal_eval(requests.get(url, params = params).text)

            except:
                return ''
            
            if 'knowledge_graph' in response.keys() and 'description' in response['knowledge_graph'].keys():
                return response['knowledge_graph']['description']
            if 'organic_results' in response.keys() and len(response['organic_results']) > 0:
                
                return '\n'.join([res['snippet'] for res in response['organic_results'][:item_num]])
            return ''
        
class SearchEngine():
    def search(self, keywords: str, model: str = "gpt-4o-mini", timeout: int = 30) -> str:
        """Initiates a chat session with DuckDuckGo AI.

        Args:
            keywords (str): The initial message or question to send to the AI.
            model (str): The model to use: "gpt-4o-mini", "claude-3-haiku", "llama-3.1-70b", "mixtral-8x7b".
                Defaults to "gpt-4o-mini".
            timeout (int): Timeout value for the HTTP client. Defaults to 30.

        Returns:
            str: The response from the AI.
        """
        ddgs = DDGS()
        response = ddgs.chat(keywords, model, timeout)
        return response



if __name__ == "__main__":
    # search_engine = GoogleSearchEngine()
    # print(search_engine.search("cell phone tower"))

    # print(SearchAPIEngine().search("Juergen Schmidhuber"))
    print(SearchEngine().search("Marvin Minsky"))