import os
import sys
import pandas as pd
import sqlite3
import logfire

# Configuration des chemins absolus depuis le dossier 'scripts'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Ajout de la racine au PYTHONPATH pour permettre l'importation du dossier schemas
sys.path.append(PROJECT_ROOT)

from schemas.models import PlayerModel, SeasonStatModel

logfire.configure()

DB_PATH = os.path.join(PROJECT_ROOT, "vector_db", "nba_stats.db")
EXCEL_PATH = os.path.join(PROJECT_ROOT, "inputs", "regular NBA.xlsx")

@logfire.instrument("Initialisation de la base SQL")
def setup_database(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE, team TEXT, age INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS season_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            gp INTEGER, w INTEGER, l INTEGER, min REAL, pts INTEGER,
            fgm INTEGER, fga INTEGER, fg_pct REAL, three_pm INTEGER, three_pa INTEGER, three_p_pct REAL,
            ftm INTEGER, fta INTEGER, ft_pct REAL, oreb INTEGER, dreb INTEGER, reb INTEGER,
            ast INTEGER, tov INTEGER, stl INTEGER, blk INTEGER, pf INTEGER, fp INTEGER,
            dd2 INTEGER, td3 INTEGER, plus_minus REAL, offrtg REAL, defrtg REAL, netrtg REAL,
            ast_pct REAL, ast_to REAL, ast_ratio REAL, oreb_pct REAL, dreb_pct REAL, reb_pct REAL,
            to_ratio REAL, efg_pct REAL, ts_pct REAL, usg_pct REAL, pace REAL, pie REAL, poss INTEGER,
            FOREIGN KEY(player_id) REFERENCES players(id)
        )
    ''')

@logfire.instrument("Ingestion globale Excel vers SQL")
def load_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    setup_database(cursor)

    df = pd.read_excel(EXCEL_PATH, sheet_name="Données NBA", header=1)
    
    col_3pm = [col for col in df.columns if isinstance(col, pd.Timestamp) or str(col).startswith('15:00')][0]
    
    rename_map = {
        'Player': 'name', 'Team': 'team', 'Age': 'age', 'GP': 'gp', 'W': 'w', 'L': 'l', 'Min': 'min',
        'PTS': 'pts', 'FGM': 'fgm', 'FGA': 'fga', 'FG%': 'fg_pct', col_3pm: 'three_pm',
        '3PA': 'three_pa', '3P%': 'three_p_pct', 'FTM': 'ftm', 'FTA': 'fta', 'FT%': 'ft_pct',
        'OREB': 'oreb', 'DREB': 'dreb', 'REB': 'reb', 'AST': 'ast', 'TOV': 'tov',
        'STL': 'stl', 'BLK': 'blk', 'PF': 'pf', 'FP': 'fp', 'DD2': 'dd2', 'TD3': 'td3',
        '+/-': 'plus_minus', 'OFFRTG': 'offrtg', 'DEFRTG': 'defrtg', 'NETRTG': 'netrtg',
        'AST%': 'ast_pct', 'AST/TO': 'ast_to', 'AST RATIO': 'ast_ratio', 'OREB%': 'oreb_pct',
        'DREB%': 'dreb_pct', 'REB%': 'reb_pct', 'TO RATIO': 'to_ratio', 'EFG%': 'efg_pct',
        'TS%': 'ts_pct', 'USG%': 'usg_pct', 'PACE': 'pace', 'PIE': 'pie', 'POSS': 'poss'
    }
    df = df.rename(columns=rename_map)

    for _, row in df.iterrows():
        if pd.isna(row['name']):
            continue
            
        try:
            player_data = PlayerModel(**row.to_dict())
            stat_data = SeasonStatModel(**row.to_dict())

            cursor.execute('''INSERT OR IGNORE INTO players (name, team, age) VALUES (?, ?, ?)''', 
                           (player_data.name, player_data.team, player_data.age))
            
            cursor.execute('SELECT id FROM players WHERE name = ?', (player_data.name,))
            player_id = cursor.fetchone()[0]

            stat_dict = stat_data.model_dump()
            stat_dict['player_id'] = player_id
            
            columns = ', '.join(stat_dict.keys())
            placeholders = ', '.join(['?'] * len(stat_dict))
            values = tuple(stat_dict.values())
            
            cursor.execute(f'''INSERT INTO season_stats ({columns}) VALUES ({placeholders})''', values)

        except Exception as e:
            logfire.warning(f"Erreur d'ingestion pour {row.get('name', 'Inconnu')}: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    load_data()