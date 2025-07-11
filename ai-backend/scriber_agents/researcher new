import logging
from typing import Any, List, Dict
from dotenv import load_dotenv
import json

from agents import Agent, Runner

load_dotenv()
logger = logging.getLogger(__name__)


class ResearchAgent:
    """Agent responsible for researching contextual information and analysis."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Research Agent with configuration."""
        self.config = config or {}

        self.agent = Agent(
            instructions="""You are a sports research agent specializing in analyzing game data, team history, and player performance. 
            Your task is to provide clear, engaging storylines and analysis that junior writers can easily understand and use.

            CRITICAL REQUIREMENTS:
            - ONLY use information that is explicitly provided in the data
            - DO NOT invent, assume, or speculate about any facts not present in the data
            - If data is missing or incomplete, acknowledge this limitation
            - Base all analysis strictly on the factual data provided
            - Do not add external knowledge or assumptions

            Focus on:
            1. Most important 3-5 storylines only (based on provided data)
            2. Historical context between teams (from provided data only)
            3. Individual player performances and impact (from provided data only)
            4. Key moments and turning points (from provided data only)
            5. Tactical and strategic insights (from provided data only)

            Guidelines:
            - Keep analysis simple and accessible for junior writers
            - Focus on what makes this match/player/team interesting based on actual data
            - Provide factual, objective analysis using only provided information
            - Highlight human interest elements that are supported by the data
            - Consider broader context and significance only if supported by the data
            - If data is insufficient, state what information is missing rather than making assumptions

            Always return clear, structured analysis that writers can immediately use, based solely on the provided data.""",
            name="ResearchAgent",
            output_type=str,
            model=self.config.get("model", "gpt-4o-mini"),
        )

        logger.info("Research Agent initialized successfully")

    async def get_storyline_from_game_data(self, game_data: dict) -> list[str]:
        logger.info("Generating storylines from game data (current match events only)")
        prompt = f"""Extract 3-5 factual storylines from this match only. Do not include anything not explicitly present in the data.
        {game_data}"""
        return await self._run_agent_prompt(prompt)

    async def get_turning_points(self, game_data: dict) -> list[str]:
        logger.info("Identifying turning points from game data")
        prompt = f"""Identify 2-3 key turning points in this match based on game-changing events (e.g., red cards, late goals).
        Use only what's present in this data:
        {game_data}"""
        return await self._run_agent_prompt(prompt)

    async def get_performance_from_player_game_data(self, player_data: dict, game_data: dict) -> list[str]:
        logger.info("Analyzing individual player performance from game data (current match events only)")
        prompt = f"""Analyze what players actually did in this match using the following:
        Game Data:
        {game_data}
        Player Data:
        {player_data}"""
        return await self._run_agent_prompt(prompt)

    async def get_history_from_team_data(self, team_data: dict) -> list[str]:
        logger.info("Analyzing historical context from team data (background information only)")
        prompt = f"""Extract 3-5 background facts about the teams using only this data:
        {team_data}"""
        return await self._run_agent_prompt(prompt)

    async def get_event_timeline(self, game_data: dict) -> list[str]:
        logger.info("Generating minute-by-minute event timeline")
        prompt = f"""Create a chronological timeline of match events with timestamps.
        Use only the following game data:
        {game_data}"""
        return await self._run_agent_prompt(prompt)

    async def get_stat_summary(self, stat_data: dict) -> list[str]:
        logger.info("Extracting statistical summary from match data")
        prompt = f"""Summarize numeric match stats (possession, shots, cards, corners, etc.) using only this data:
        {stat_data}"""
        return await self._run_agent_prompt(prompt)

    async def get_best_and_worst_moments(self, game_data: dict) -> Dict[str, str]:
        logger.info("Finding best and worst moments in match")
        prompt = f"""From this match data, provide:
        - best_moment (e.g. a decisive goal)
        - worst_moment (e.g. a missed penalty)
        Output JSON with 'best_moment' and 'worst_moment' keys.
        {game_data}"""
        try:
            result = await Runner.run(self.agent, prompt)
            return json.loads(result.final_output)
        except Exception as e:
            logger.error(f"Error generating best/worst moments: {e}")
            return {"best_moment": "Unavailable", "worst_moment": "Unavailable"}

    async def get_missed_chances(self, game_data: dict) -> list[str]:
        logger.info("Identifying missed chances from match data")
        prompt = f"""List all missed chances or penalties that had potential impact on the match based on the following data:
        {game_data}"""
        return await self._run_agent_prompt(prompt)

    async def get_formations_from_lineup_data(self, lineup_data: dict) -> list[str]:
        logger.info("Extracting team formations from lineup data")
        prompt = f"""Identify and return team formations (e.g., 4-3-3, 3-5-2) for both teams based on this lineup data:
        {lineup_data}"""
        return await self._run_agent_prompt(prompt)

    async def _run_agent_prompt(self, prompt: str) -> list[str]:
        try:
            result = await Runner.run(self.agent, prompt)
            try:
                parsed = json.loads(result.final_output)
                if isinstance(parsed, list):
                    return [str(line).strip() for line in parsed if line]
            except Exception:
                return [line.strip() for line in result.final_output.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error running prompt: {e}")
            return ["Analysis based on available data"]

