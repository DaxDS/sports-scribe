import asyncio
import logging
from agents.research_agent import ResearchAgent
from data_collector import DataCollector  # Assuming this module provides data fetching

logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize data collector and research agent
    collector = DataCollector()
    agent = ResearchAgent()

    # Fetch test data
    game_data = await collector.get_game_data(match_id="test_match_001")
    player_data = await collector.get_player_data(match_id="test_match_001")
    team_data = await collector.get_team_data(match_id="test_match_001")
    stat_data = await collector.get_stat_data(match_id="test_match_001")
    lineup_data = await collector.get_lineup_data(match_id="test_match_001")

    # Run all analysis methods
    print("\n--- Storylines ---")
    print(await agent.get_storyline_from_game_data(game_data))

    print("\n--- Turning Points ---")
    print(await agent.get_turning_points(game_data))

    print("\n--- Player Performance ---")
    print(await agent.get_performance_from_player_game_data(player_data, game_data))

    print("\n--- Historical Context ---")
    print(await agent.get_history_from_team_data(team_data))

    print("\n--- Event Timeline ---")
    print(await agent.get_event_timeline(game_data))

    print("\n--- Statistical Summary ---")
    print(await agent.get_stat_summary(stat_data))

    print("\n--- Best and Worst Moments ---")
    print(await agent.get_best_and_worst_moments(game_data))

    print("\n--- Missed Chances ---")
    print(await agent.get_missed_chances(game_data))

    print("\n--- Formations ---")
    print(await agent.get_formations_from_lineup_data(lineup_data))


if __name__ == "__main__":
    asyncio.run(main())
