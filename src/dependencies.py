import aiohttp, asyncio, json
from pydantic import BaseModel
from typing import AsyncGenerator, List, Optional, Callable


class LLMMessage(BaseModel):
    role:              str = "assistant"
    content:           str
    content_delta:     Optional[str] = None
    response_finished: bool = False

class LLMProviderConfig(BaseModel):
    api_key:             str|List[str]
    base_url:            str
    provider:            str = "openai"
    primary_model:       str
    secondary_model:     str
    enable_key_rotation: bool = False

class LLMConnector:
    def __init__(
        self,
        providers: List[LLMProviderConfig],
        provider_priority: Optional[List[str]] = None
    ) -> None:
        self.providers = providers
        
        for config in providers:
            provider_name = config.provider
            if isinstance(config.api_key, str):
                config.api_key = [config.api_key]
            setattr(self, provider_name, config)
        
        # Set provider priority
        if provider_priority:
            self.provider_priority = [p for p in provider_priority if hasattr(self, p)]
        else:
            self.provider_priority = [p.provider for p in providers]
        if not self.provider_priority:
            raise ValueError("No valid providers configured in priority list")
        
        # Create key rotation counters and locks
        self.key_counters = {p: 0 for p in self.provider_priority}
        self.key_locks = {p: asyncio.Lock() for p in self.provider_priority}
    
    async def _get_rotated_key(self, provider_name: str) -> str:
        """Get the next API key in rotation with thread-safety"""
        config: LLMProviderConfig = getattr(self, provider_name)
        if not config.enable_key_rotation or len(config.api_key) <= 1:
            return config.api_key[0]
        async with self.key_locks[provider_name]:
            key_index = self.key_counters[provider_name]
            api_key = config.api_key[key_index]
            self.key_counters[provider_name] = (key_index + 1) % len(config.api_key)
        return api_key
    
    async def _stream_openai(self, provider_name: str, model: str, messages: List[dict], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI-compatible providers"""
        config = getattr(self, provider_name)
        api_key = await self._get_rotated_key(provider_name)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 2049),
            "temperature": kwargs.get("temperature", 0.5),
        }
        async with aiohttp.ClientSession() as session:
            async with self.session.post(
                f"{config.base_url}/chat/completions", 
                headers=headers, 
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Error from {provider_name}: {response.status}, {error_text}")
                    
                # Process SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        data = json.loads(line[6:])
                        delta = data['choices'][0]['delta']
                        if 'content' in delta and delta['content'] is not None:
                            yield delta['content']
        
    async def _stream_anthropic(self, model: str, system: Optional[str], messages: List[dict], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic"""
        api_key = await self._get_rotated_key("anthropic")
        config: LLMProviderConfig = getattr(self, "anthropic")
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2049),
            "temperature": kwargs.get("temperature", 0.5),
            "stream": True
        }
        
        if system:
            payload["system"] = system
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config.base_url}/messages", 
                headers=headers, 
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Error from anthropic: {response.status}, {error_text}")
                    
                # Process SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        try:
                            data = json.loads(line[6:])
                            if data.get('type') == 'content_block_delta':
                                if 'delta' in data and 'text' in data['delta']:
                                    yield data['delta']['text']
                        except json.JSONDecodeError:
                            continue
             
    async def _try_next_provider(self, current_index: int) -> str:
        """Try next provider in the priority list"""
        if current_index >= len(self.provider_priority) - 1:
            raise Exception("All providers failed")
        return self.provider_priority[current_index + 1]
                
    async def stream(
        self, 
        system: str, 
        messages: List[dict], 
        max_tokens: int = 2049, 
        temperature: float=0.5,
        provider: Optional[str] = None,
        use_secondary_model: bool = False
    ):
        output = []
        current_index = 0
        current_provider = provider if provider in self.provider_priority else self.provider_priority[0]
        
        while True:
            try:
                config: LLMProviderConfig = getattr(self, current_provider)
                model = config.secondary_model if use_secondary_model else config.primary_model 
                
                if current_provider == "anthropic":
                    stream_gen = self._stream_anthropic(model, system, messages, max_tokens=max_tokens, temperature=temperature)
                else:
                    formatted_messages = [{"role": "system", "content": system}] + messages if system else messages
                    stream_gen = self._stream_openai(current_provider, model, formatted_messages, max_tokens=max_tokens, temperature=temperature)
                
                async for text in stream_gen:
                    output.append(text)
                    yield LLMMessage(content_delta=text, content="".join(output))
                break
            except Exception as e:
                if provider:
                    raise Exception(f"Specified provider {provider} failed: {str(e)}")
                try:
                    current_provider = await self._try_next_provider(current_index)
                    current_index += 1
                    output = []
                except Exception as e:
                    raise Exception(f"All providers failed. Last error: {str(e)}")
        
        yield LLMMessage(content="".join(output), content_delta="", response_finished=True)
        
    async def generate(
        self,
        messages: List[dict], 
        extractor_function: Optional[Callable] = None, 
        system: Optional[str] = None, 
        max_tokens: int = 2049, 
        enable_cache: bool = False,
        temperature: float = 0.3,
        provider: Optional[str] = None,
        use_primary_model: bool = False
    ):
        current_index = 0
        current_provider = provider if provider in self.provider_priority else self.provider_priority[0]
    
        while True:
            try:
                api_key = await self._get_rotated_key(current_provider)
                config: LLMProviderConfig = getattr(self, current_provider)
                model = config.primary_model if use_primary_model else config.secondary_model
                if current_provider == "anthropic":
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    }
                    
                    if enable_cache:
                        headers["anthropic-beta"] = "prompt-caching-2024-07-31"
                        endpoint = f"{config.base_url}/beta/prompt_caching/messages"
                    else:
                        endpoint = f"{config.base_url}/messages"
                    
                    payload = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                    
                    if system:
                        payload["system"] = system
                        
                    async with aiohttp.ClientSession() as session:
                        async with session.post(endpoint, headers=headers, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"Error from anthropic: {response.status}, {error_text}")
                            data = await response.json()
                            response_text = data['content'][0]['text']
                else:
                    # Format messages for OpenAI-compatible APIs
                    if system:
                        formatted_messages = [{"role": "system", "content": system}] + messages
                    else:
                        formatted_messages = messages
                    
                    # Process multi-part content
                    # _messages = []
                    # for message in formatted_messages:
                    #     if isinstance(message['content'], list):
                    #         content = "\n\n".join([msg['text'] for msg in message['content'] if 'text' in msg])
                    #         _messages.append({"role": message['role'], "content": content})
                    #     elif isinstance(message['content'], str):
                    #         _messages.append({"role": message['role'], "content": message['content']})
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                    
                    payload = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": formatted_messages,
                    }
                    
                    endpoint = f"{config.base_url}/chat/completions"
                    async with aiohttp.ClientSession() as session:
                        async with session.post(endpoint, headers=headers, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"Error from {current_provider}: {response.status}, {error_text}")
                            
                            data = await response.json()
                            response_text = data['choices'][0]['message']['content']
                break
            except Exception as e:
                if provider:
                    raise Exception(f"Specified provider {provider} failed: {str(e)}")
                try:
                    current_provider = await self._try_next_provider(current_index)
                    current_index += 1
                except Exception as e:
                    raise Exception(f"All providers failed. Last error: {str(e)}")
                    
        return extractor_function(response_text) if extractor_function else response_text

