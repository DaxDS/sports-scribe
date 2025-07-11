"""
Streamlined Pipeline Orchestrator.

This module coordinates the flow between different agents in the SportsScribe pipeline:
Data Collector → Research → Writer
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, List

from .data_collector import DataCollectorAgent
from .researcher import ResearchAgent
from .writer import WriterAgent
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class AgentPipeline:
    """Streamlined pipeline orchestrating data flow between agents."""

    def __init__(self):
        """Initialize the pipeline with all required agents."""
        # Get configuration from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not self.rapidapi_key:
            raise ValueError("RAPIDAPI_KEY environment variable is required")
        
        # Create config dict for agents
        config = {
            "openai_api_key": self.openai_api_key,
            "rapidapi_key": self.rapidapi_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
        # Initialize all agents
        self.collector = DataCollectorAgent(config)
        self.researcher = ResearchAgent(config)
        self.writer = WriterAgent(config)
        
        logger.info("AgentPipeline initialized successfully")

    async def generate_game_recap(self, game_id: str) -> Dict[str, Any]:
        """Generate a complete game recap article.
        
        Pipeline: Data Collection → Research → Writer
        """
        pipeline_start_time = datetime.now()
        logger.info(f"[PIPELINE] Starting game recap generation for game: {game_id}")
        
        try:
            # Step 1: Data Collection
            logger.info(f"[PIPELINE] Step 1: Collecting game data for {game_id}")
            raw_game_data = await self._collect_game_data(game_id)
            if not raw_game_data:
                raise ValueError(f"Failed to collect data for game {game_id}")
            
            # Check if data collection resulted in errors
            if raw_game_data.get("errors") and len(raw_game_data.get("errors", [])) > 0:
                logger.warning(f"[PIPELINE] Data collection had errors: {raw_game_data['errors']}")
                if raw_game_data.get("results", 0) == 0:
                    raise ValueError(f"No data available for game {game_id}: {raw_game_data['errors']}")
            
            # Log raw data information
            logger.info(f"[PIPELINE-DATA] Raw game data collected:")
            logger.info(f"[PIPELINE-DATA]   Type: {type(raw_game_data)}")
            logger.info(f"[PIPELINE-DATA]   Keys: {list(raw_game_data.keys()) if isinstance(raw_game_data, dict) else 'Not a dict'}")
            if isinstance(raw_game_data, dict):
                logger.info(f"[PIPELINE-DATA]   Response count: {raw_game_data.get('response', [])}")
                logger.info(f"[PIPELINE-DATA]   Errors: {raw_game_data.get('errors', [])}")
                logger.info(f"[PIPELINE-DATA]   Results: {raw_game_data.get('results', 0)}")
            
            logger.info(f"[PIPELINE] Raw game data collected successfully")
            
            # Step 1.5: Extract team and player information
            logger.info(f"[PIPELINE] Step 1.5: Extracting team and player information")
            team_info = self.extract_team_info(raw_game_data)
            player_info = self.extract_player_info(raw_game_data)
            
            # Log extracted information
            logger.info(f"[PIPELINE-DATA] Team info extracted:")
            logger.info(f"[PIPELINE-DATA]   Type: {type(team_info)}")
            if isinstance(team_info, dict) and "error" not in team_info:
                home_team = team_info.get("home_team", {}).get("name", "Unknown")
                away_team = team_info.get("away_team", {}).get("name", "Unknown")
                logger.info(f"[PIPELINE-DATA]   Teams: {home_team} vs {away_team}")
                logger.info(f"[PIPELINE-DATA]   League: {team_info.get('league', {}).get('name', 'Unknown')}")
            else:
                logger.warning(f"[PIPELINE-DATA]   Team info error: {team_info.get('error', 'Unknown error')}")
            
            logger.info(f"[PIPELINE-DATA] Player info extracted:")
            logger.info(f"[PIPELINE-DATA]   Type: {type(player_info)}")
            if isinstance(player_info, dict) and "error" not in player_info:
                total_players = len(player_info.get("all_players", {}))
                key_players = len(player_info.get("key_players", []))
                logger.info(f"[PIPELINE-DATA]   Total players: {total_players}")
                logger.info(f"[PIPELINE-DATA]   Key players: {key_players}")
            else:
                logger.warning(f"[PIPELINE-DATA]   Player info error: {player_info.get('error', 'Unknown error')}")
            
            logger.info(f"[PIPELINE] Team and player information extracted successfully")
            
            # Step 1.6: Collect enhanced team and player data using data collector
            logger.info(f"[PIPELINE] Step 1.6: Collecting enhanced team and player data")
            enhanced_team_data = await self.collect_enhanced_team_data(team_info)
            season = None
            try:
                response_list = raw_game_data.get("response", [])
                if response_list and isinstance(response_list, list):
                    season = response_list[0].get("league", {}).get("season")
            except Exception as e:
                logger.warning(f"[PIPELINE] Failed to extract season: {e}")
            enhanced_player_data = await self.collect_enhanced_player_data(player_info, season)
            
            # Log enhanced data collection
            logger.info(f"[PIPELINE-DATA] Enhanced team data collected:")
            logger.info(f"[PIPELINE-DATA]   Type: {type(enhanced_team_data)}")
            if isinstance(enhanced_team_data, dict) and "error" not in enhanced_team_data:
                enhanced_data = enhanced_team_data.get("enhanced_data", {})
                home_detailed = "home_team_detailed" in enhanced_data
                away_detailed = "away_team_detailed" in enhanced_data
                logger.info(f"[PIPELINE-DATA]   Home team detailed: {home_detailed}")
                logger.info(f"[PIPELINE-DATA]   Away team detailed: {away_detailed}")
            else:
                logger.warning(f"[PIPELINE-DATA]   Enhanced team data error: {enhanced_team_data.get('error', 'Unknown error')}")
            
            logger.info(f"[PIPELINE-DATA] Enhanced player data collected:")
            logger.info(f"[PIPELINE-DATA]   Type: {type(enhanced_player_data)}")
            if isinstance(enhanced_player_data, dict) and "error" not in enhanced_player_data:
                enhanced_key_players = len(enhanced_player_data.get("enhanced_key_players", []))
                sample_players = len(enhanced_player_data.get("sample_players_detailed", []))
                logger.info(f"[PIPELINE-DATA]   Enhanced key players: {enhanced_key_players}")
                logger.info(f"[PIPELINE-DATA]   Sample players detailed: {sample_players}")
            else:
                logger.warning(f"[PIPELINE-DATA]   Enhanced player data error: {enhanced_player_data.get('error', 'Unknown error')}")
            
            logger.info(f"[PIPELINE] Enhanced team and player data collected successfully")
            
            # Step 2: Research and generate storylines
            logger.info(f"[PIPELINE] Step 2: Conducting research and generating storylines")
            
            # Step 2.1: Analyze game data for storylines
            logger.info(f"[PIPELINE] Step 2.1: Analyzing game data for storylines")
            game_analysis = await self.researcher.get_storyline_from_game_data(raw_game_data)
            logger.info(f"[PIPELINE-DATA] Game analysis storylines: {len(game_analysis) if isinstance(game_analysis, list) else 'Not a list'}")

            # Step 2.2: Analyze historical context between teams
            logger.info(f"[PIPELINE] Step 2.2: Analyzing historical context between teams")
            historical_context = await self.researcher.get_history_from_team_data(enhanced_team_data)
            logger.info(f"[PIPELINE-DATA] Historical context storylines: {len(historical_context) if isinstance(historical_context, list) else 'Not a list'}")

            # Step 2.3: Analyze individual player performances
            logger.info(f"[PIPELINE] Step 2.3: Analyzing individual player performances")
            player_performance_analysis = await self.researcher.get_performance_from_player_game_data(enhanced_player_data, raw_game_data)
            logger.info(f"[PIPELINE-DATA] Player performance storylines: {len(player_performance_analysis) if isinstance(player_performance_analysis, list) else 'Not a list'}")
            
            # Combine all research data into a comprehensive structure
            # NOTE: Keep storylines separate from historical context to avoid confusion
            comprehensive_research_data = {
                "game_analysis": game_analysis,  # Current match events only
                "historical_context": historical_context,  # Background information only
                "player_performance": player_performance_analysis,  # Current match player events only
            }
            
            # Log research data information
            logger.info(f"[PIPELINE-DATA] Comprehensive research data:")
            logger.info(f"[PIPELINE-DATA]   Type: {type(comprehensive_research_data)}")
            logger.info(f"[PIPELINE-DATA]   Keys: {list(comprehensive_research_data.keys())}")
            logger.info(f"[PIPELINE-DATA]   Game analysis storylines: {len(game_analysis)}")
            logger.info(f"[PIPELINE-DATA]   Historical context: {len(historical_context)}")
            logger.info(f"[PIPELINE-DATA]   Player performance: {len(player_performance_analysis)}")
            
            logger.info(f"[PIPELINE] Research completed, generated {len(game_analysis)} game storylines, {len(historical_context)} historical context items, {len(player_performance_analysis)} player performance items")
            
            # Step 3: Generate article content
            logger.info(f"[PIPELINE] Step 3: Generating article content")
            
            # Prepare data for writer
            game_info = raw_game_data
            research_for_writer = comprehensive_research_data
            
            # Log the data being passed to writer for debugging
            logger.info(f"[PIPELINE-DEBUG] Data passed to writer:")
            logger.info(f"[PIPELINE-DEBUG]   game_info type: {type(game_info)}, keys: {list(game_info.keys()) if isinstance(game_info, dict) else 'Not a dict'}")
            logger.info(f"[PIPELINE-DEBUG]   research type: {type(research_for_writer)}, keys: {list(research_for_writer.keys()) if isinstance(research_for_writer, dict) else 'Not a dict'}")
            
            # Generate article using the writer agent
            article_content = await self.writer.generate_game_recap(
                game_info, research_for_writer
            )
            
            # Log article content information
            logger.info(f"[PIPELINE-DATA] Generated article:")
            logger.info(f"[PIPELINE-DATA]   Type: {type(article_content)}")
            logger.info(f"[PIPELINE-DATA]   Length: {len(article_content) if isinstance(article_content, str) else 'Not a string'}")
            if isinstance(article_content, str):
                logger.info(f"[PIPELINE-DATA]   Preview: {article_content[:200]}...")
            
            logger.info(f"[PIPELINE] Article content generated successfully")
            
            # Step 4: Return results
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.info(f"[PIPELINE] Game recap generation completed in {pipeline_duration:.2f} seconds")
            
            return {
                "success": True,
                "game_id": game_id,
                "article_type": "game_recap",
                "content": article_content,
                # "storylines": game_analysis,  # Only current match events for storylines
                # "team_info": enhanced_team_data,
                # "player_info": enhanced_player_data,
                # "research_data": comprehensive_research_data,
                # "historical_context": historical_context,
                # "player_performance_analysis": player_performance_analysis,
                # "metadata": {
                #     "generated_at": datetime.now().isoformat(),
                #     "pipeline_duration": pipeline_duration,
                #     "data_sources": ["rapidapi_football"],
                #     "model_used": self.model,
                #     "format_manager_used": False,
                #     "team_info_extracted": "error" not in team_info,
                #     "player_info_extracted": "error" not in player_info,
                #     "enhanced_team_data_collected": "error" not in enhanced_team_data,
                #     "enhanced_player_data_collected": "error" not in enhanced_player_data,
                #     "historical_context_analyzed": "error" not in historical_context,
                #     "player_performance_analyzed": "error" not in player_performance_analysis,
                #     "comprehensive_storylines_generated": len(game_analysis) > 0
                # }
            }
            
        except Exception as e:
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.error(f"[PIPELINE] Error generating game recap for {game_id} after {pipeline_duration:.2f} seconds: {str(e)}")
            return {
                "success": False,
                "game_id": game_id,
                "error": str(e),
                "research_data": {
                    "game_analysis": None,
                    "historical_context": None,
                    "player_performance": None,
                    "storylines": [],
                    "team_info": None,
                    "player_info": None
                },
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "pipeline_duration": pipeline_duration,
                    "data_sources": ["rapidapi_football"],
                    "model_used": self.model,
                    "error_occurred": True,
                    "error_step": "pipeline_execution"
                }
            }

    async def _collect_game_data(self, game_id: str) -> Dict[str, Any]:
        """Collect game data using the data collector agent."""
        try:
            logger.info(f"[PIPELINE] Collecting game data for {game_id}")
            data = await self.collector.collect_game_data(game_id)
            logger.info(f"[PIPELINE] Game data collected successfully")
            return data
        except Exception as e:
            logger.error(f"[PIPELINE] Failed to collect game data: {e}")
            
            # Return a structured error response instead of raising
            return {
                "get": f"game data for fixture {game_id}",
                "parameters": {"fixture_id": game_id},
                "errors": [f"Failed to collect game data: {str(e)}"],
                "results": 0,
                "paging": {"current": 1, "total": 1},
                "response": []
            }

    def extract_team_info(self, raw_game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract team information from raw game data.
        
        Args:
            raw_game_data: Raw game data from API response
            
        Returns:
            Dictionary containing extracted team information
        """
        try:
            logger.info("[PIPELINE] Extracting team information from raw game data")
            
            # Extract response data
            response_list = raw_game_data.get("response", [])
            if not response_list:
                logger.warning("[PIPELINE] No response data found in raw_game_data")
                return {"error": "No response data available"}
            
            fixture_data = response_list[0]
            teams = fixture_data.get("teams", {})
            
            # Extract home team info
            home_team = teams.get("home", {})
            home_team_info = {
                "id": home_team.get("id"),
                "name": home_team.get("name"),
                "logo": home_team.get("logo"),
                "winner": home_team.get("winner")
            }
            
            # Extract away team info
            away_team = teams.get("away", {})
            away_team_info = {
                "id": away_team.get("id"),
                "name": away_team.get("name"),
                "logo": away_team.get("logo"),
                "winner": away_team.get("winner")
            }
            
            # Extract league info
            league = fixture_data.get("league", {})
            league_info = {
                "id": league.get("id"),
                "name": league.get("name"),
                "country": league.get("country"),
                "logo": league.get("logo"),
                "flag": league.get("flag"),
                "season": league.get("season"),
                "round": league.get("round")
            }
            
            # Extract lineup information if available
            lineups = fixture_data.get("lineups", [])
            home_lineup = None
            away_lineup = None
            
            for lineup in lineups:
                team_id = lineup.get("team", {}).get("id")
                if team_id == home_team_info["id"]:
                    home_lineup = {
                        "formation": lineup.get("formation"),
                        "coach": lineup.get("coach", {}).get("name"),
                        "startXI": lineup.get("startXI", []),
                        "substitutes": lineup.get("substitutes", [])
                    }
                elif team_id == away_team_info["id"]:
                    away_lineup = {
                        "formation": lineup.get("formation"),
                        "coach": lineup.get("coach", {}).get("name"),
                        "startXI": lineup.get("startXI", []),
                        "substitutes": lineup.get("substitutes", [])
                    }
            
            team_info = {
                "home_team": home_team_info,
                "away_team": away_team_info,
                "league": league_info,
                "season": league_info.get("season"),
                "home_lineup": home_lineup,
                "away_lineup": away_lineup
            }
            
            logger.info(f"[PIPELINE] Successfully extracted team info for {home_team_info['name']} vs {away_team_info['name']}")
            return team_info
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error extracting team info: {e}")
            return {"error": f"Failed to extract team info: {str(e)}"}

    def extract_player_info(self, raw_game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract player information from raw game data.
        
        Args:
            raw_game_data: Raw game data from API response
            
        Returns:
            Dictionary containing extracted player information
        """
        try:
            logger.info("[PIPELINE] Extracting player information from raw game data")
            
            # Extract response data
            response_list = raw_game_data.get("response", [])
            if not response_list:
                logger.warning("[PIPELINE] No response data found in raw_game_data")
                return {"error": "No response data available"}
            
            fixture_data = response_list[0]
            
            # Extract events (goals, cards, substitutions)
            events = fixture_data.get("events", [])
            player_events = {}
            
            for event in events:
                player = event.get("player", {})
                player_id = player.get("id")
                player_name = player.get("name")
                
                if player_id and player_name:
                    if player_id not in player_events:
                        player_events[player_id] = {
                            "id": player_id,
                            "name": player_name,
                            "team": event.get("team", {}).get("name"),
                            "team_id": event.get("team", {}).get("id"),
                            "events": []
                        }
                    
                    player_events[player_id]["events"].append({
                        "type": event.get("type"),
                        "detail": event.get("detail"),
                        "time": event.get("time", {}).get("elapsed"),
                        "assist": event.get("assist", {}).get("name") if event.get("assist") else None
                    })
            
            # Extract lineup information for all players
            lineups = fixture_data.get("lineups", [])
            all_players = {}
            
            for lineup in lineups:
                team_name = lineup.get("team", {}).get("name")
                team_id = lineup.get("team", {}).get("id")
                
                # Process starting XI
                for player_data in lineup.get("startXI", []):
                    player = player_data.get("player", {})
                    player_id = player.get("id")
                    if player_id:
                        all_players[player_id] = {
                            "id": player_id,
                            "name": player.get("name"),
                            "number": player.get("number"),
                            "position": player.get("pos"),
                            "team": team_name,
                            "team_id": team_id,
                            "status": "started",
                            "formation_position": player.get("grid")
                        }
                
                # Process substitutes
                for player_data in lineup.get("substitutes", []):
                    player = player_data.get("player", {})
                    player_id = player.get("id")
                    if player_id:
                        all_players[player_id] = {
                            "id": player_id,
                            "name": player.get("name"),
                            "number": player.get("number"),
                            "position": player.get("pos"),
                            "team": team_name,
                            "team_id": team_id,
                            "status": "substitute",
                            "formation_position": None
                        }
            
            # Merge event data with player data
            for player_id, player_data in all_players.items():
                if player_id in player_events:
                    player_data["match_events"] = player_events[player_id]["events"]
                else:
                    player_data["match_events"] = []
            
            # Separate players by team
            home_team_id = fixture_data.get("teams", {}).get("home", {}).get("id")
            away_team_id = fixture_data.get("teams", {}).get("away", {}).get("id")
            
            home_players = {pid: pdata for pid, pdata in all_players.items() 
                          if pdata.get("team_id") == home_team_id}
            away_players = {pid: pdata for pid, pdata in all_players.items() 
                          if pdata.get("team_id") == away_team_id}
            
            player_info = {
                "home_players": home_players,
                "away_players": away_players,
                "all_players": all_players,
                "key_players": self._identify_key_players(all_players, events)
            }
            
            logger.info(f"[PIPELINE] Successfully extracted player info for {len(all_players)} players")
            return player_info
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error extracting player info: {e}")
            return {"error": f"Failed to extract player info: {str(e)}"}

    def _identify_key_players(self, all_players: Dict[str, Any], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key players based on match events.
        
        Args:
            all_players: Dictionary of all players
            events: List of match events
            
        Returns:
            List of key players with their achievements
        """
        key_players = []
        
        for event in events:
            if event.get("type") in ["Goal", "Card"]:
                player = event.get("player", {})
                player_id = player.get("id")
                
                if player_id and player_id in all_players:
                    player_data = all_players[player_id].copy()
                    player_data["key_achievement"] = {
                        "type": event.get("type"),
                        "detail": event.get("detail"),
                        "time": event.get("time", {}).get("elapsed")
                    }
                    key_players.append(player_data)
        
        return key_players

    async def collect_enhanced_team_data(self, team_info: Dict[str, Any]) -> Dict[str, Any]:
        """Collect enhanced team data using data collector.
        
        Args:
            team_info: Basic team information extracted from game data
            
        Returns:
            Dictionary containing enhanced team data
        """
        try:
            logger.info("[PIPELINE] Collecting enhanced team data")
            
            enhanced_team_data = {
                "home_team": team_info.get("home_team", {}),
                "away_team": team_info.get("away_team", {}),
                "league": team_info.get("league", {}),
                "home_lineup": team_info.get("home_lineup", {}),
                "away_lineup": team_info.get("away_lineup", {}),
                "enhanced_data": {}
            }
            
            # Collect detailed data for home team
            home_team_id = team_info.get("home_team", {}).get("id")
            if home_team_id:
                try:
                    logger.info(f"[PIPELINE] Collecting detailed data for home team {home_team_id}")
                    home_team_detailed = await self.collector.collect_team_data(str(home_team_id))
                    enhanced_team_data["enhanced_data"]["home_team_detailed"] = home_team_detailed
                    logger.info(f"[PIPELINE] Successfully collected home team detailed data")
                except Exception as e:
                    logger.warning(f"[PIPELINE] Failed to collect home team detailed data: {e}")
                    enhanced_team_data["enhanced_data"]["home_team_detailed"] = {"error": str(e)}
            
            # Collect detailed data for away team
            away_team_id = team_info.get("away_team", {}).get("id")
            if away_team_id:
                try:
                    logger.info(f"[PIPELINE] Collecting detailed data for away team {away_team_id}")
                    away_team_detailed = await self.collector.collect_team_data(str(away_team_id))
                    enhanced_team_data["enhanced_data"]["away_team_detailed"] = away_team_detailed
                    logger.info(f"[PIPELINE] Successfully collected away team detailed data")
                except Exception as e:
                    logger.warning(f"[PIPELINE] Failed to collect away team detailed data: {e}")
                    enhanced_team_data["enhanced_data"]["away_team_detailed"] = {"error": str(e)}
            
            logger.info("[PIPELINE] Enhanced team data collection completed")
            return enhanced_team_data
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error collecting enhanced team data: {e}")
            return {"error": f"Failed to collect enhanced team data: {str(e)}"}

    async def collect_enhanced_player_data(self, player_info: Dict[str, Any], season: str) -> Dict[str, Any]:
        """Collect enhanced player data using data collector.
        
        Args:
            player_info: Basic player information extracted from game data
            
        Returns:
            Dictionary containing enhanced player data
        """
        try:
            logger.info("[PIPELINE] Collecting enhanced player data")
            
            enhanced_player_data = {
                "home_players": player_info.get("home_players", {}),
                "away_players": player_info.get("away_players", {}),
                "all_players": player_info.get("all_players", {}),
                "key_players": player_info.get("key_players", []),
                "enhanced_data": {}
            }
            
            # Collect detailed data for key players (limit to top 5 to avoid too many API calls)
            key_players = player_info.get("key_players", [])
            enhanced_key_players = []
            
            if not season:
                logger.warning("[PIPELINE] Season not found, cannot collect enhanced player data.")
                return {"error": "Season not available in raw game data"}

            for i, player in enumerate(key_players[:5]):  # Limit to top 5 key players
                player_id = player.get("id")
                if player_id:
                    try:
                        logger.info(f"[PIPELINE] Collecting detailed data for key player {player_id} ({player.get('name', 'Unknown')})")
                        player_detailed = await self.collector.collect_player_data(str(player_id), str(season))
                        
                        enhanced_player = player.copy()
                        enhanced_player["detailed_data"] = player_detailed
                        enhanced_key_players.append(enhanced_player)
                        
                        logger.info(f"[PIPELINE] Successfully collected detailed data for player {player_id}")
                    except Exception as e:
                        logger.warning(f"[PIPELINE] Failed to collect detailed data for player {player_id}: {e}")
                        enhanced_player = player.copy()
                        enhanced_player["detailed_data"] = {"error": str(e)}
                        enhanced_key_players.append(enhanced_player)
            
            enhanced_player_data["enhanced_key_players"] = enhanced_key_players
            
            # Collect detailed data for a few sample players from each team (for context)
            home_players = list(player_info.get("home_players", {}).values())
            away_players = list(player_info.get("away_players", {}).values())
            
            # Collect data for 2-3 players from each team
            sample_players = []
            
            # Sample from home team
            for player in home_players[:2]:
                player_id = player.get("id")
                if player_id:
                    try:
                        logger.info(f"[PIPELINE] Collecting sample data for home player {player_id}")
                        player_detailed = await self.collector.collect_player_data(str(player_id), str(season))
                        
                        sample_player = player.copy()
                        sample_player["detailed_data"] = player_detailed
                        sample_players.append(sample_player)
                    except Exception as e:
                        logger.warning(f"[PIPELINE] Failed to collect sample data for home player {player_id}: {e}")
            
            # Sample from away team
            for player in away_players[:2]:
                player_id = player.get("id")
                if player_id:
                    try:
                        logger.info(f"[PIPELINE] Collecting sample data for away player {player_id}")
                        player_detailed = await self.collector.collect_player_data(str(player_id), str(season))
                        
                        sample_player = player.copy()
                        sample_player["detailed_data"] = player_detailed
                        sample_players.append(sample_player)
                    except Exception as e:
                        logger.warning(f"[PIPELINE] Failed to collect sample data for away player {player_id}: {e}")
            
            enhanced_player_data["sample_players_detailed"] = sample_players
            
            logger.info(f"[PIPELINE] Enhanced player data collection completed. Key players: {len(enhanced_key_players)}, Sample players: {len(sample_players)}")
            return enhanced_player_data
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error collecting enhanced player data: {e}")
            return {"error": f"Failed to collect enhanced player data: {str(e)}"}

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline and its agents."""
        return {
            "pipeline_status": "operational",
            "agents": {
                "data_collector": "initialized",
                "researcher": "initialized",
                "writer": "initialized"
            },
            "configuration": {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
            "data_flow": "Data Collector → Research → Writer",
            "timestamp": datetime.now().isoformat()
        }


# Legacy ArticlePipeline class for backward compatibility
class ArticlePipeline(AgentPipeline):
    """Legacy pipeline class - now inherits from AgentPipeline."""
    
    def __init__(self):
        """Initialize the legacy pipeline."""
        super().__init__()
        logger.info("Legacy ArticlePipeline initialized (using new AgentPipeline)") 