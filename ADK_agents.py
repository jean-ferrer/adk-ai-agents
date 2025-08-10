### Imports ###

from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import google.genai as genai
from dotenv import load_dotenv

from typing import Any, Dict, List, Union
from itertools import product
import pandas as pd
import numpy as np
import requests
import asyncio
import pprint
import json
import io
import os
import re
import zipfile
import tarfile
import PyPDF2



### === Definições Iniciais === ###

# Constantes dos agentes
APP_NAME = "full_data_pipeline_app"
USER_ID = "dev_user_01"
SESSION_ID = "session_01"
GEMINI_MODEL = "gemini-2.0-flash"    # [ 'gemini-2.0-flash' | 'gemini-2.0-flash-lite' | 'gemini-2.5-flash' | 'gemini-2.5-flash-lite' ]

# Outras constantes
URL = "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2024.zip"    # URL para download dos dados
DATA_DIR = "DATA"                                                                       # diretório local onde os dados brutos serão baixados e extraídos
TIME = 1                                                                                # pausa em segundos entre cada ação para evitar erros de "quota exceeded" da API (limite de requisições por minuto)
MAX_ITERATIONS = 3                                                                      # máximo de loops da pipeline agêntica

# Chaves de Estados da sessão
DATA_WORKSPACE = {}
STATE_ENGINEERING_SUMMARY = "engineering_summary"
STATE_PERFORMANCE_METRICS = "performance_metrics"
STATE_HYPERPARAMETERS = "hyperparameters"
STATE_CRITIQUE = "critique_output"
REENGINEER_SIGNAL = "REVISE DATA ENGINEERING"
TUNE_HYPERPARAMETERS_SIGNAL = "REVISE HYPERPARAMETER TUNING"

# Query do usuário
INITIAL_QUERY = (
    f"Verifique os dados contidos na pasta '{DATA_DIR}' e encontre o arquivo principal referente às escolas. "
    "Utilize os dicionários de dados dos datasets (se existirem). "
    "O objetivo é prever se uma escola possui internet. "
#    "O objetivo é prever se uma escola possui água potável. "
    "Selecione colunas relevantes (como a localização e infraestrutura da escola) para construir o modelo."
    "Para o workflow, utilize somente as ferramentas previamente dadas."
)

# Carrega variáveis de ambiente
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")

# print("Modelos disponíveis que suportam 'generateContent':")
# print("-------------------------------------------------")

# client = genai.Client(api_key=GOOGLE_API_KEY)

# for model in client.models.list():
#     print(f"Nome da API: {model.name}")
#     print(f"  Nome de Exibição: {model.display_name}")
#     print(f"  Descrição: {model.description}\n")



### Dowload e Extração dos Dados ###

def download_and_extract(url: str, data_dir: str):
    """
    Baixa um arquivo de uma URL, salva-o em um diretório específico e o extrai.

    A função verifica se o arquivo já foi baixado e se o conteúdo já foi
    extraído antes de executar as operações, evitando trabalho redundante.

    Args:
        url: A URL do arquivo a ser baixado.
        data_dir: O diretório para salvar o arquivo e extrair seu conteúdo.
    """
    # Garante que o diretório de destino exista.
    print(f"--- Garantindo que o diretório '{data_dir}' exista. ---")
    os.makedirs(data_dir, exist_ok=True)

    # Define o caminho de salvamento do arquivo dentro do diretório de dados.
    filename = os.path.basename(url)
    archive_path = os.path.join(data_dir, filename)

    # Verifica se o arquivo já existe para evitar um novo download.
    if not os.path.exists(archive_path):
        print(f"--- Baixando arquivo de {url} para {archive_path} ---")
        try:
            response = requests.get(url, stream=True)
            # Lança uma exceção para respostas com erro (ex: 404, 500).
            response.raise_for_status()
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("--- Download concluído com sucesso. ---")
        except requests.exceptions.RequestException as e:
            print(f"Erro ao baixar o arquivo: {e}")
            return # Interrompe a execução se o download falhar.
    else:
        print(f"--- O arquivo '{archive_path}' já existe. Pulando o download. ---")

    # Verifica se o conteúdo já foi extraído.
    # Esta verificação assume que o arquivo .zip contém uma pasta principal
    # com o mesmo nome do arquivo (ex: 'ml-latest-small.zip' -> 'ml-latest-small/').
    extracted_folder_name = os.path.splitext(filename)[0]
    expected_extracted_path = os.path.join(data_dir, extracted_folder_name)

    if os.path.exists(expected_extracted_path):
        print(f"--- O conteúdo já parece ter sido extraído em '{data_dir}'. Pulando a extração. ---")
    else:
        # Extrai o arquivo baixado para o mesmo diretório.
        print(f"--- Extraindo {archive_path} para {data_dir} ---")
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
            elif archive_path.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(path=data_dir)
            else:
                print(f"Formato de arquivo não suportado para extração: {archive_path}")
                return

            print(f"--- Arquivo extraído com sucesso para '{data_dir}'. ---")

        except (zipfile.BadZipFile, tarfile.ReadError) as e:
            print(f"Erro ao extrair o arquivo: {e}")

print("Iniciando o processo de download e extração...")
download_and_extract(url=URL, data_dir=DATA_DIR)
print("\nProcesso finalizado.")
print(f"Verifique a pasta '{DATA_DIR}' para ver os resultados.")



### Ferramentas (Tools) dos AI Agents ###

def list_project_files(start_path: str) -> dict:
    """
    Lists all folders, subfolders, and their files within a directory.

    Args:
        start_path: The directory to start listing from.

    Returns:
        A dictionary with the status and a string representing the file tree.
    """
    if '..' in start_path:
        return {"status": "error", "message": "Path cannot contain '..'. Access is restricted."}
    try:
        tree_string = ""
        for root, dirs, files in os.walk(start_path):
            if any(d in root for d in ['__pycache__', '.venv', 'env', '.git']):
                continue
            level = root.replace(start_path, '').count(os.sep)
            indent = " " * 4 * level
            tree_string += f"{indent}{os.path.basename(root)}/\n"
            sub_indent = " " * 4 * (level + 1)
            for f in files:
                tree_string += f"{sub_indent}{f}\n"
        print(f"--- Tool: Listing files in {start_path} ---")
        return {"status": "success", "file_tree": tree_string or "No files or directories found."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def read_file(file_path: str) -> dict:
    """
    Reads the content of a file, handling both text and PDF formats.

    Args:
        file_path: The path to the file.

    Returns:
        A dictionary with the status and the content of the file.
    """
    max_chars = 1000 # the maximum number of characters to return from the file content.

    file_extension = file_path.split('.')[-1].lower()
    try:
        if file_extension == 'pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        if len(content) > max_chars:
            content = content[:max_chars] + "..."  # Truncate content

        print(f"--- Tool: Reading file {file_path} ---")
        return {"status": "success", "file_path": file_path, "content": content}
    except Exception as e:
        return {"status": "error", "message": f"Error reading file {file_path}: {e}"}


def inspect_file_structure(file_path: str, num_rows: int, header_row: int) -> dict:
    """
    Previews the top of a file to identify its structure (header, columns, delimiter).

    Args:
        file_path: The path to the dataset file (CSV or Excel).
        num_rows: The number of rows to preview. If None, defaults to 10.
        header_row: The 0-indexed row presumed to be the header. If None, defaults to 0.
    """
    try:
        # --- Internal Defaults ---
        if num_rows is None:
            num_rows = 10
        if header_row is None:
            header_row = 0

        extension = os.path.splitext(file_path)[-1].lower()
        preview_args = {'header': None, 'nrows': num_rows}
        column_args = {'header': header_row}

        if extension == '.csv':
            df_preview = pd.read_csv(file_path, **preview_args, sep=None, engine='python', encoding='latin1')
            df_cols = pd.read_csv(file_path, **column_args, nrows=0, encoding='latin1')
        elif extension in ['.xls', '.xlsx']:
            df_preview = pd.read_excel(file_path, **preview_args)
            df_cols = pd.read_excel(file_path, **column_args)
        else:
            return {"status": "error", "message": f"Unsupported file type: '{extension}'."}

        print(f"--- Tool: Inspecting file structure for {file_path} ---")
        return {
            "status": "success",
            "file_path": file_path,
            "header_preview": df_preview.to_string(),
            "column_names": df_cols.columns.tolist()
        }
    except Exception as e:
        return {"status": "error", "message": f"Error inspecting file {file_path}: {e}"}


def query_data_dictionary(file_path: str, filter_values: List[str]) -> Dict[str, Union[str, int, List[Dict[str, Any]]]]:
    """
    Queries a data dictionary file to retrieve specific rows based on a filter.

    This function reads a CSV or Excel file and scans every cell. It returns the
    complete rows where at least one cell contains any of the strings in `filter_values`.
    The search is case-insensitive. All data in the returned rows is truncated to a 
    maximum of 50 characters per cell.

    Args:
        file_path: The path to the data file (CSV or Excel).
        filter_values: A list of string values to search for anywhere in the file.

    Returns:
        A dictionary with the operation's status. On success, it includes the
        retrieved data as a list of dictionaries, where each dictionary represents a
        matching and truncated row.
    """
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        # Determine file type and read it into a DataFrame
        extension = os.path.splitext(file_path)[-1].lower()
        if extension == '.csv':
            # Using 'latin1' encoding for broader compatibility with datasets
            df = pd.read_csv(file_path, encoding='latin1')
        elif extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            return {"status": "error", "message": f"Unsupported file type: '{extension}'."}

        # Create a single regex pattern to search for any of the filter_values, case-insensitively
        # re.escape handles special characters in the filter values
        pattern = r'|'.join(re.escape(val) for val in filter_values)
        
        # Search all columns for the pattern.
        # First, convert all data to string to apply string operations uniformly.
        # This returns a boolean DataFrame of the same shape as `df`.
        mask = df.astype(str).apply(lambda x: x.str.contains(pattern, case=False, na=False))
        
        # Keep rows where at least one column (cell) matched the pattern
        filtered_df = df[mask.any(axis=1)]
        # Replace all NaN values with an empty string
        cleaned_df = filtered_df.fillna('')
        # Truncate all values in the filtered DataFrame to 50 characters
        truncated_df = cleaned_df.applymap(lambda x: str(x)[:50] if pd.notna(x) else x)
        # Convert the final DataFrame to a list of dictionaries
        records = truncated_df.to_dict(orient='records')
        
        print(f"--- Tool: Searched {file_path} and found {len(records)} matching rows. ---")
        return {"status": "success", "file_path": file_path, "data": records}

    except Exception as e:
        return {"status": "error", "message": f"An error occurred while searching {file_path}: {e}"}


def load_dataset(file_name: str, header_row: int, use_columns: List[str], delimiter: str) -> dict:
    """
    Loads data from a CSV or Excel file into a DataFrame in the workspace.

    Args:
        file_name: The path of the file to load.
        header_row: The 0-indexed row containing column names. If None, defaults to 0.
        use_columns: A list of column names to load. If None, all columns are loaded.
        delimiter: The character for separating values in a CSV. If None, defaults to ','.
    """
    try:
        # --- Internal Defaults ---
        if header_row is None:
            header_row = 0
        if delimiter is None:
            delimiter = ','

        extension = os.path.splitext(file_name)[-1].lower()
        if extension == '.csv':
            df = pd.read_csv(file_name, header=header_row, usecols=use_columns, sep=delimiter,
                             low_memory=False, encoding='latin1')
        elif extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_name, header=header_row, usecols=use_columns)
        else:
            return {"status": "error", "message": "Invalid file type. Must be 'csv' or 'xlsx'."}

        df_key = f"df_{os.path.basename(file_name).split('.')[0]}"
        DATA_WORKSPACE[df_key] = df
        print(f"--- Tool: load_dataset successful. Stored under key: {df_key} ---")
        
        return {
            "status": "success",
            "df_key": df_key,
            "rows_loaded": len(df),
            "columns_loaded": len(df.columns),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
 

def preview_dataset(df_key: str) -> dict:
    """
    Previews the first n rows of a DataFrame from the workspace.

    Args:
        df_key: The key of the DataFrame in the workspace.

    Returns:
        A dictionary with status and a string representation of the DataFrame's head.
    """
    n = 10 # Number of rows to be seen

    if df_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"DataFrame key '{df_key}' not found."}
    
    df = DATA_WORKSPACE[df_key]
    return {"status": "success", "df_key": df_key, "preview": df.head(n).to_string()}


def dataset_info(df_key: str) -> dict:
    """
    Provides technical information about a DataFrame (columns, types, non-null counts).

    Args:
        df_key: The key of the DataFrame in the workspace.

    Returns:
        A dictionary with status and the DataFrame's info as a string.
    """
    if df_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"DataFrame key '{df_key}' not found."}
    
    df = DATA_WORKSPACE[df_key]
    buffer = io.StringIO()
    df.info(buf=buffer)
    return {"status": "success", "df_key": df_key, "info": buffer.getvalue()}


def describe_dataset(df_key: str) -> dict:
    """
    Provides descriptive statistics for a DataFrame in the workspace.
    Includes statistics for both numeric and object/categorical columns.

    Args:
        df_key: The key of the DataFrame in the workspace.

    Returns:
        A dictionary with the status and a string representation of the described DataFrame.
    """
    if df_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"DataFrame key '{df_key}' not found."}
    
    try:
        df = DATA_WORKSPACE[df_key]
        # include='all' ensures that statistics for all column types are generated.
        description = df.describe(include='all').to_string()
        print(f"--- Tool: Describing dataset {df_key} ---")
        return {"status": "success", "df_key": df_key, "description": description}
    except Exception as e:
        return {"status": "error", "message": f"Error describing DataFrame {df_key}: {e}"}


def clean_dataset(df_key: str) -> dict:
    """
    Cleans a DataFrame by filling NaN values based on column type and removing duplicates.

    - For numeric columns with more than 2 unique values, NaNs are filled with the mean.
    - For object/categorical columns or numeric columns with 2 or fewer unique values
      (i.e., boolean-like), NaNs are filled with the mode.
    - Duplicate rows are removed after filling NaNs.
    
    The operation modifies the DataFrame in place.

    Args:
        df_key: The key of the DataFrame in DATA_WORKSPACE to clean.

    Returns:
        A dictionary with status and statistics about the cleaning process.
    """
    if df_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"DataFrame key '{df_key}' not found."}

    try:
        df = DATA_WORKSPACE[df_key]
        rows_before = len(df)
        nan_count_before = int(df.isnull().sum().sum())
        imputation_details = {}

        # Iterate through each column to fill NaN values
        for col in df.columns:
            if df[col].isnull().any():
                # Heuristic: Treat as numerical if dtype is numeric and it's not boolean-like
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2:
                    fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                    imputation_details[col] = {"method": "mean", "value": fill_value}
                # Treat as categorical/boolean for object types or low-cardinality numerics
                else:
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    imputation_details[col] = {"method": "mode", "value": fill_value}

        # Handle duplicates after NaN imputation
        rows_after_fill = len(df)
        df.drop_duplicates(inplace=True)
        rows_after_dedup = len(df)
        
        DATA_WORKSPACE[df_key] = df
        
        return {
            "status": "success",
            "df_key": df_key,
            "rows_before": rows_before,
            "rows_after": rows_after_dedup,
            "nan_values_filled": nan_count_before,
            "duplicates_removed": rows_after_fill - rows_after_dedup,
            "imputation_details": imputation_details
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def convert_to_categorical(df_key: str, columns_to_convert: list[str]) -> dict:
    """
    Converts specified columns in a DataFrame to the 'category' dtype. Modifies in place.

    Args:
        df_key: The key of the DataFrame to modify.
        columns_to_convert: A list of column names to convert.

    Returns:
        A dictionary confirming the status and listing the converted columns.
    """
    if df_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"DataFrame key '{df_key}' not found."}
    
    try:
        df = DATA_WORKSPACE[df_key]
        converted = []
        not_found = []
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = df[col].astype('category')
                converted.append(col)
            else:
                not_found.append(col)
        
        DATA_WORKSPACE[df_key] = df
        return {
            "status": "success",
            "df_key": df_key,
            "converted_columns": converted,
            "columns_not_found": not_found
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

def convert_to_int(df_key: str, columns_to_convert: list[str]) -> dict:
    """
    Converts specified columns in a DataFrame to the integer dtype. Modifies in place.

    Note: This will fail if a column contains non-numeric values or NaNs.
    The calling code should handle such potential errors.

    Args:
        df_key: The key of the DataFrame to modify in DATA_WORKSPACE.
        columns_to_convert: A list of column names to convert.

    Returns:
        A dictionary confirming the status and listing the converted columns.
    """
    if df_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"DataFrame key '{df_key}' not found."}

    try:
        df = DATA_WORKSPACE[df_key]
        converted = []
        not_found = []
        
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(int)
                converted.append(col)
            else:
                not_found.append(col)
        
        DATA_WORKSPACE[df_key] = df
        return {
            "status": "success",
            "df_key": df_key,
            "converted_columns": converted,
            "columns_not_found": not_found
        }
    except Exception as e:
        # This will catch errors like trying to convert a column with NaN or non-numeric strings
        return {"status": "error", "message": str(e)}


def split_features_target(df_key: str, target_column: str) -> dict:
    """
    Splits a DataFrame into features (X) and target (y). Stores them in the workspace.

    Args:
        df_key: The key of the DataFrame to split.
        target_column: The name of the target column (y).

    Returns:
        A dictionary with status and the new keys for features (X) and target (y).
    """
    if df_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"DataFrame key '{df_key}' not found."}
    
    df = DATA_WORKSPACE[df_key]
    if target_column not in df.columns:
        return {"status": "error", "message": f"Target column '{target_column}' not in DataFrame."}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_key = f"X_{df_key}"
    y_key = f"y_{df_key}"
    
    DATA_WORKSPACE[X_key] = X
    DATA_WORKSPACE[y_key] = y
    
    return {"status": "success", "features_key": X_key, "target_key": y_key}


def train_test_split_data_for_classifier(X_key: str, y_key: str, test_size: float, random_state: int) -> dict:
    """
    Splits features (X) and target (y) into training and testing sets.
    Automatically sets the `stratify` parameter to y, so that the target is not unbalanced.

    Args:
        X_key: The workspace key for the features DataFrame (X).
        y_key: The workspace key for the target Series (y).
        test_size: Proportion for the test split.
        random_state: Seed for reproducibility.

    Returns:
        A dictionary with status and the keys for X_train, X_test, y_train, and y_test.
    """
    if X_key not in DATA_WORKSPACE or y_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"Feature key '{X_key}' or target key '{y_key}' not found."}
        
    X = DATA_WORKSPACE[X_key]
    y = DATA_WORKSPACE[y_key]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y) # stratify para que o target não seja desbalanceado
    
    keys = {
        "X_train": f"{X_key}_train", "X_test": f"{X_key}_test",
        "y_train": f"{y_key}_train", "y_test": f"{y_key}_test"
    }
    
    DATA_WORKSPACE[keys["X_train"]] = X_train
    DATA_WORKSPACE[keys["X_test"]] = X_test
    DATA_WORKSPACE[keys["y_train"]] = y_train
    DATA_WORKSPACE[keys["y_test"]] = y_test
    
    return {"status": "success", "data_keys": keys}


def train_test_split_data_for_regressor(X_key: str, y_key: str, test_size: float, random_state: int) -> dict:
    """
    Splits features (X) and target (y) into training and testing sets.

    Args:
        X_key: The workspace key for the features DataFrame (X).
        y_key: The workspace key for the target Series (y).
        test_size: Proportion for the test split.
        random_state: Seed for reproducibility.

    Returns:
        A dictionary with status and the keys for X_train, X_test, y_train, and y_test.
    """
    if X_key not in DATA_WORKSPACE or y_key not in DATA_WORKSPACE:
        return {"status": "error", "message": f"Feature key '{X_key}' or target key '{y_key}' not found."}
        
    X = DATA_WORKSPACE[X_key]
    y = DATA_WORKSPACE[y_key]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    
    keys = {
        "X_train": f"{X_key}_train", "X_test": f"{X_key}_test",
        "y_train": f"{y_key}_train", "y_test": f"{y_key}_test"
    }
    
    DATA_WORKSPACE[keys["X_train"]] = X_train
    DATA_WORKSPACE[keys["X_test"]] = X_test
    DATA_WORKSPACE[keys["y_train"]] = y_train
    DATA_WORKSPACE[keys["y_test"]] = y_test
    
    return {"status": "success", "data_keys": keys}
    

def train_xgboost_model(X_train_key: str, y_train_key: str, model_type: str) -> dict:
    """
    Trains an XGBoost model (Classifier or Regressor) with a fixed set of
    hyperparameters and stores it in the workspace.

    Args:
        X_train_key: The workspace key for the training features (X_train).
        y_train_key: The workspace key for the training target (y_train).
        model_type: The type of model to train, either 'classifier' or 'regressor'.

    Returns:
        A dictionary containing the status, the key for the trained model, and the
        hyperparameters that were used.
    """
    # Check if the specified training data exists in the workspace
    if X_train_key not in DATA_WORKSPACE or y_train_key not in DATA_WORKSPACE:
        return {"status": "error", "message": "Training data keys not found in workspace."}

    try:
        X_train = DATA_WORKSPACE[X_train_key]
        y_train = DATA_WORKSPACE[y_train_key]

        # Define the fixed set of hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'enable_categorical': True,
            'random_state': 42 # Added for reproducibility
        }

        # Select and instantiate the model based on the model_type parameter
        if model_type.lower() == 'classifier':
            model = XGBClassifier(**params)
            model_key = "xgb_classifier_model"
        elif model_type.lower() == 'regressor':
            model = XGBRegressor(**params)
            model_key = "xgb_regressor_model"
        else:
            return {
                "status": "error",
                "message": "Invalid model_type. Please choose 'classifier' or 'regressor'."
            }

        # Train the selected model
        model.fit(X_train, y_train)

        # Store the trained model in the workspace
        DATA_WORKSPACE[model_key] = model

        return {
            "status": "success",
            "model_key": model_key,
            "hyperparameters_used": params
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def evaluate_classifier_performance(model_key: str, X_test_key: str, y_test_key: str) -> dict:
    """
    Evaluates a classifier model using precision, recall, and F1-score.

    Args:
        model_key: The workspace key of the trained classifier model.
        X_test_key: The workspace key of the test features (X_test).
        y_test_key: The workspace key of the true test target values (y_test).

    Returns:
        A dictionary containing the status and performance metrics.
    """
    if model_key not in DATA_WORKSPACE or X_test_key not in DATA_WORKSPACE or y_test_key not in DATA_WORKSPACE:
        return {"status": "error", "message": "Model or test data keys not found in workspace."}
        
    try:
        model = DATA_WORKSPACE[model_key]
        X_test = DATA_WORKSPACE[X_test_key]
        y_test = DATA_WORKSPACE[y_test_key]
        y_pred = model.predict(X_test)
        
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        return {
            "status": "success",
            "metrics": {"Precision": precision, "Recall": recall, "F1-Score": f1}
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to evaluate classifier: {e}"}


def evaluate_regressor_performance(model_key: str, X_test_key: str, y_test_key: str) -> dict:
    """
    Evaluates a regressor model using MAE, RMSE, and R-squared.

    Args:
        model_key: The workspace key of the trained regressor model.
        X_test_key: The workspace key of the test features (X_test).
        y_test_key: The workspace key of the true test target values (y_test).

    Returns:
        A dictionary containing the status and performance metrics.
    """
    if model_key not in DATA_WORKSPACE or X_test_key not in DATA_WORKSPACE or y_test_key not in DATA_WORKSPACE:
        return {"status": "error", "message": "Model or test data keys not found in workspace."}

    try:
        model = DATA_WORKSPACE[model_key]
        X_test = DATA_WORKSPACE[X_test_key]
        y_test = DATA_WORKSPACE[y_test_key]
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return {
            "status": "success",
            "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2}
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to evaluate regressor: {e}"}


def hyperparameter_search_xgboost(X_train_key: str, y_train_key: str, model_type: str) -> dict:
    """
    Performs a hyperparameter grid search for an XGBoost model using cross-validation.

    Args:
        X_train_key: The workspace key for the training features (X_train).
        y_train_key: The workspace key for the training target (y_train).
        model_type: The type of model to tune, either 'classifier' or 'regressor'.

    Returns:
        A dictionary with the status, the key for the best model found, and the best parameters.
    """
    if X_train_key not in DATA_WORKSPACE or y_train_key not in DATA_WORKSPACE:
        return {"status": "error", "message": "Training data keys not found in workspace."}
    if model_type not in ['classifier', 'regressor']:
        return {"status": "error", "message": "model_type must be 'classifier' or 'regressor'."}

    try:
        X_train = DATA_WORKSPACE[X_train_key]
        y_train = DATA_WORKSPACE[y_train_key]

        # Convert data to DMatrix for XGBoost efficiency
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

        # Define hyperparameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'eta': np.logspace(start=0.01, stop=0.3, num=5),
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8]
        }
        
        # Configure search based on model type
        if model_type == 'regressor':
            objective = 'reg:squarederror'
            eval_metric = 'rmse'
        else: # classifier
            if y_train.nunique() == 2:
                objective = 'binary:logistic'
                eval_metric = 'logloss'
            else:
                objective = 'multi:softmax'
                eval_metric = 'mlogloss'

        # Placeholders for results
        best_score = float("inf")
        best_params_combo = {}
        
        # Create all combinations for the grid search
        search_space = list(product(*param_grid.values()))

        print(f"--- Tool: Starting XGBoost hyperparameter search for {model_type} ({len(search_space)} combinations) ---")

        for params_tuple in search_space:
            params = dict(zip(param_grid.keys(), params_tuple))
            params['objective'] = objective
            
            if model_type == 'classifier' and y_train.nunique() > 2:
                 params['num_class'] = y_train.nunique()

            # Execute cross-validation
            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=500,
                nfold=3,
                metrics=eval_metric,
                early_stopping_rounds=25,
                verbose_eval=False
            )
            
            # Get the best score from this CV run (lowest error)
            current_score = cv_results[f'test-{eval_metric}-mean'].min()
            
            # Update best score and params if current run is better
            if current_score < best_score:
                best_score = current_score
                best_params_combo = params
                # Find the optimal number of boosting rounds
                best_iteration = cv_results[f'test-{eval_metric}-mean'].idxmin()
                best_params_combo['n_estimators'] = best_iteration

        # Train the final model with the absolute best parameters
        final_model_params = {k: v for k, v in best_params_combo.items() if k not in ['objective', 'num_class']}
        
        if model_type == 'regressor':
            model = XGBRegressor(**final_model_params, objective=objective, enable_categorical=True)
        else:
            model = XGBClassifier(**final_model_params, objective=objective, enable_categorical=True)
            
        model.fit(X_train, y_train)

        model_key = f"xgb_{model_type}_tuned_model"
        DATA_WORKSPACE[model_key] = model
        
        return {
            "status": "success", 
            "model_key": model_key,
            "best_params_found": best_params_combo
        }

    except Exception as e:
        return {"status": "error", "message": f"Hyperparameter search failed: {e}"}


def save_model_and_metadata(model_key: str, X_train_key: str, hyperparameters: Dict[str, Any], model_type: str) -> dict:
    """
    Saves the trained model and its metadata (columns, hyperparameters).
    The output folder is set internally to 'trained_model_artifacts'.

    Args:
        model_key: The workspace key for the trained model.
        X_train_key: The workspace key for the training features (X_train).
        hyperparameters: A dictionary of hyperparameters used for training.
        model_type: The type of model ('classifier' or 'regressor').

    Returns:
        A dictionary with the status and paths to the saved files.
    """
    # --- Internal Defaults ---
    output_folder = "trained_model_artifacts"

    if model_key not in DATA_WORKSPACE or X_train_key not in DATA_WORKSPACE:
        return {"status": "error", "message": "Model or training data key not found in workspace."}

    try:
        # Create the output directory if it doesn't already exist
        os.makedirs(output_folder, exist_ok=True)

        # 1. Save the XGBoost model using its native method
        model = DATA_WORKSPACE[model_key]
        model_path = os.path.join(output_folder, f"{model_key}.json")
        model.save_model(model_path)

        # 2. Prepare and save the metadata
        X_train = DATA_WORKSPACE[X_train_key]
        # Ensure all hyperparameter values are JSON serializable
        serializable_hyperparameters = {k: (v.item() if hasattr(v, 'item') else v) for k, v in hyperparameters.items()}
        
        metadata = {
            "model_type": model_type,
            "feature_columns": X_train.columns.tolist(),
            "hyperparameters": serializable_hyperparameters,
        }
        metadata_path = os.path.join(output_folder, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"--- Tool: Saved model to {model_path} and metadata to {metadata_path} ---")
        return {
            "status": "success",
            "model_path": model_path,
            "metadata_path": metadata_path
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def exit_loop(tool_context: ToolContext) -> dict:
    """
    Signals the main agent loop to stop iterating.

    Args:
        tool_context: The context object provided by the ADK framework.

    Returns:
        A dictionary confirming that the exit signal has been sent.
    """
    print(f"--- [Tool Call] exit_loop activated by {tool_context.agent_name} ---")
    tool_context.actions.escalate = True
    return {"status": "success", "message": "Exit signal sent to the main loop."}


# Lista das ferramentas disponíveis para o agente de Data Engineering
ENGINEERING_TOOLS = [list_project_files, inspect_file_structure, query_data_dictionary, read_file]

# Lista das ferramentas disponíveis para o agente de Data Science
SCIENCE_TOOLS = [load_dataset, dataset_info, describe_dataset, preview_dataset, clean_dataset, convert_to_categorical, convert_to_int, 
                 split_features_target, train_test_split_data_for_classifier, train_test_split_data_for_regressor, train_xgboost_model,
                 evaluate_classifier_performance, evaluate_regressor_performance, hyperparameter_search_xgboost, save_model_and_metadata]

# Lista das ferramentas disponíveis para o agente de Avaliação
CRITIQUE_TOOLS = [exit_loop]



### Definição dos Agentes ###

# 1. O Agente "Engenheiro de Dados"
data_engineer_agent = LlmAgent(
    name="DataEngineerAgent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are a highly efficient Data Engineer AI. Your goal is to logically identify and prepare the features for a machine learning model. You must follow these steps in order:

    **1. Explore and Inspect:**
        - Use `list_project_files` to identify the primary dataset and the data dictionary file.
        - Use `inspect_file_structure` on the **primary dataset** to get the exact list of all available column names and determine the `header_row` and `delimiter`.

    **2. Make a Preliminary Selection:**
        - From the full list of column names you just obtained, make a **preliminary selection** of approximately 10 columns that seem most relevant for the ML model based on their names alone.

    **3. Validate Selection with Data Dictionary:**
        - Use the `query_data_dictionary` tool on the **data dictionary file** to retrieve the descriptions **only for the ~10 columns chosen in the preliminary selection**.
          This allows you to confirm they are suitable.
        - If needed, use the `read_file` function to read relevant information about the dataset from text or PDF files.

    **4. Finalize Features and Prepare Output:**
        - Review the descriptions returned by `query_data_dictionary`. Based on their meanings, confirm or revise your list to create the final set of features.
        - The session state key '{STATE_CRITIQUE}' may contain feedback. If it contains the signal "{REENGINEER_SIGNAL}", you **MUST** choose a **different combination of features**.
        - Your final output for this turn **MUST** be a single, valid JSON object that the next agent will use to call the `load_dataset` function.
        - It must contain the keys `file_name`, `header_row`, `use_columns`, and optionally `delimiter`.
        - Example: {{"file_name": "path/to/data.csv", "header_row": 0, "use_columns": ["col1", "col2", "col3"], "delimiter": ";"}}
    """,
    tools=ENGINEERING_TOOLS,
    output_key=STATE_ENGINEERING_SUMMARY
)

# 2. O Agente "Cientista de Dados"
data_scientist_agent = LlmAgent(
    name="DataScientistAgent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are a methodical Data Scientist AI. Your task is to preprocess data, train a model, evaluate it, and save the final artifacts. You must follow these steps in order:

    **1. Load Data & Initial Analysis:**
        - Get the data loading parameters from the session state key '{STATE_ENGINEERING_SUMMARY}'.
        - Call the `load_dataset` tool. This returns a dictionary containing the `df_key`, which you **MUST** use to reference the dataset in all subsequent steps.
        - **Crucial Analysis & Verification Step:** After loading, use `dataset_info`, `preview_dataset`, and `describe_dataset` on the `df_key` to understand the data's structure,
          content, and distribution before proceeding.
        - **Evaluate columns for a high proportion of missing values (NaNs). Columns with excessive NaNs are candidates for removal, as listwise deletion could discard a substantial
          portion of the dataset, while imputation might introduce significant bias.**
        
    **2. Preprocess and Split:**
        - Use the `clean_dataset` tool on the `df_key`.
        - Use `convert_to_categorical` on the `df_key` for columns that are not numerical.
        - Use `convert_to_int` on the `df_key` for columns that are supposed to be integers and NOT floats (e.g.: 0.0 and 1.0).
        - Use `split_features_target` to separate features (X) and target (y).
        - Based on the target variable, use either `train_test_split_data_for_classifier` or `train_test_split_data_for_regressor`.

    **3. Train Model:**
        - **Decision Point:** Check the session state key '{STATE_CRITIQUE}' for a decision from the previous loop.
        - **If the decision was "{TUNE_HYPERPARAMETERS_SIGNAL}":** You **MUST** call the `hyperparameter_search_xgboost` tool. Determine the `model_type` ('classifier' or 'regressor')
          based on the split function you used.
        - **Otherwise (first run):** You **MUST** call the `train_xgboost_model` tool. Determine the `model_type` ('classifier' or 'regressor') based on which split function you used
          in the previous step. This tool uses a fixed set of default hyperparameters internally.
        - **Crucially, you MUST capture the output of this step, as it contains the `model_key` and the `hyperparameters_used` or `best_params_found` needed for the next steps.**

    **4. Evaluate Model:**
        - Use the `model_key` from the previous step to call either `evaluate_classifier_performance` or `evaluate_regressor_performance`.
        - Capture the resulting performance metrics dictionary.
        
    **5. Save Artifacts:**
        - After training and evaluation, you **MUST** save the results by calling the `save_model_and_metadata` tool.
        - Provide the following arguments:
            - `model_key`: The key of the model you just trained.
            - `X_train_key`: The key for the training features (e.g., 'X_df_matriz_train').
            - `hyperparameters`: The dictionary of hyperparameters you captured from the training step (either `hyperparameters_used` or `best_params_found`).
            - `model_type`: The type of model you trained ('classifier' or 'regressor').

    **6. Final Output:**
        - Your final output for this turn **MUST** be the complete dictionary of performance metrics returned by the evaluation tool in Step 4.
    """,
    tools=SCIENCE_TOOLS,
    output_key=STATE_PERFORMANCE_METRICS
)

# 3. O Agente "Avaliador"
critique_agent = LlmAgent(
    name="CritiqueAgent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are a decisive Machine Learning Model Critic. Your role is to analyze model performance and determine the next action with structured output. You must follow these steps in order:

    **1. Review Performance:**
        - Analyze the performance dictionary from the session state key '{STATE_PERFORMANCE_METRICS}'.
        - First, identify the primary metric ('F1-Score' or 'R-squared').
        - Apply the following logic to make your decision:
          - if score >= 0.8: Success!
          - elif 0.8 > score >= 0.6: Moderate performance. Trigger hyperparameter re-tuning.
          - elif score < 0.6: Poor performance. Signal feature re-engineering.

    **2. Take Action:**
        - **In the SUCCESS case:** Your ONLY action is to call the `exit_loop` tool. Do NOT output any JSON.
        - **For re-tuning:** Your output MUST be a single JSON object: {{"decision": "{TUNE_HYPERPARAMETERS_SIGNAL}", "reason": "Performance is moderate, initiating a full hyperparameter search."}}
        - **For re-engineering:** Your output MUST be a single JSON object: {{"decision": "{REENGINEER_SIGNAL}", "reason": "Feature selection seems inadequate."}}
    """,
    tools=CRITIQUE_TOOLS,
    output_key=STATE_CRITIQUE
)

# 4. O Agente "Orquestrador"
main_pipeline_agent = LoopAgent(
    name="MainPipelineAgent",
    sub_agents=[
        data_engineer_agent,
        data_scientist_agent,
        critique_agent
    ],
    max_iterations=MAX_ITERATIONS # Limite de loops da pipeline
)



### Pipeline ###

async def run_pipeline():
    """Configures and runs the complete agent pipeline."""
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=main_pipeline_agent, app_name=APP_NAME, session_service=session_service)

    print(f"--- STARTING PIPELINE WITH QUERY ---\n'{INITIAL_QUERY}'\n")
    content = types.Content(role='user', parts=[types.Part(text=INITIAL_QUERY)])
    
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            # Skip empty events
            if not event.content or not event.content.parts:
                continue

            processed_tool_part = False

            for part in event.content.parts:
                # 1. Check for a tool call (code the agent wants to run)
                if hasattr(part, 'executable_code') and part.executable_code:
                    print(f"\n>> {event.author} is calling a tool:")
                    print("```python")
                    print(part.executable_code.code)
                    print("```")
                    processed_tool_part = True

                # 2. Check for the result of a tool call
                elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                    output_str = pprint.pformat(part.code_execution_result.output)
                    print(f"\n>> Tool result for {event.author}:")
                    print(output_str)
                    processed_tool_part = True

            # 3. If it wasn't a tool part, it's likely a "thought" from a sub-agent.
            # The logic for accumulating a final_response from the main agent has been removed.
            if not processed_tool_part:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_content = part.text.strip()
                        # We only print the "thoughts" of the sub-agents, not the main orchestrator.
                        if text_content and event.author != main_pipeline_agent.name:
                            print(f"\n>> {event.author} is thinking...\n   {text_content}")
            
            # Pause for TIME seconds after processing each event to respect rate limits.
            await asyncio.sleep(TIME)

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---\n")
        import traceback
        traceback.print_exc()

    # The final print for 'final_response' has been removed.
    print("\n--- PIPELINE FINISHED ---")


if __name__ == "__main__":
    # Verifica se a variável API Key do Gemini NÃO existe
    if not GOOGLE_API_KEY:
        # Se não existir, avisa e ENCERRA o programa
        print("ERRO: A variável de ambiente GOOGLE_API_KEY não foi encontrada.")
        exit() # Encerra o script aqui

    # Inicializa a pipeline
    asyncio.run(run_pipeline())      # para arquivos .py
    # await run_pipeline()           # para arquivos .ipynb