import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop_duplicates().copy()
    return df

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de Modelo de Detección de Fraude")
    parser.add_argument("--data_path", type=str, default="creditcard.csv", help="Ruta al dataset")
    parser.add_argument("--c_param", type=float, default=1.0, help="Inverso de la fuerza de regularización (C)")
    parser.add_argument("--max_iter", type=int, default=1000, help="Máximo de iteraciones")
    parser.add_argument("--class_weight", type=str, default="None", help="Pesos de clases (None o balanced)")
    parser.add_argument("--random_state", type=int, default=42, help="Semilla aleatoria para reproducirlo")

    args = parser.parse_args()
    
    cw = None if args.class_weight.lower() == "none" else args.class_weight

    # Apuntamos al servidor de MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Credit_Card_Fraud_Detection_Experiment")

    run_name = f"LR_C{args.c_param}_cw_{args.class_weight}"
    with mlflow.start_run(run_name=run_name):
        # Registramos los hiperparámetros
        mlflow.log_param("C", args.c_param)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("random_state", args.random_state)
        
        # Cargamos los datos
        print(f"Cargando datos desde {args.data_path}...")
        df = load_data(args.data_path)
        X = df.drop('Class', axis=1)
        y = df['Class']

        #Set de entrenamiento y pruebas
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state, stratify=y
        )
        
        # Para prevenir el data leakage, escalamos 'Time' y 'Amount' con StandardScaler
        scaler_time = StandardScaler()
        X_train.loc[:, 'Time'] = scaler_time.fit_transform(X_train[['Time']]).flatten()
        X_test.loc[:, 'Time'] = scaler_time.transform(X_test[['Time']]).flatten()
        
        scaler_amount = StandardScaler()
        X_train.loc[:, 'Amount'] = scaler_amount.fit_transform(X_train[['Amount']]).flatten()
        X_test.loc[:, 'Amount'] = scaler_amount.transform(X_test[['Amount']]).flatten()
        
        # Entrenando el modelo
        print(f"Entrenando modelo (C={args.c_param}, max_iter={args.max_iter}, class_weight={args.class_weight})...")
        model = LogisticRegression(
            C=args.c_param, 
            max_iter=args.max_iter, 
            class_weight=cw, 
            random_state=args.random_state
        )
        model.fit(X_train, y_train)
        
        # Obtenemos las métricas de evaluación
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        # Registramos las métricas en MLFlow
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        
        # Artefacto: Matriz de confusión
        cm_path = "confusion_matrix.png"
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión (C={args.c_param}, cw={args.class_weight})")
        plt.ylabel("Real")
        plt.xlabel("Predicción")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        
        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)
        
        #Registramos el modelo entrenado
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Finalizado -> ROC-AUC: {roc_auc:.4f} | Recall: {recall:.4f} | Precisión: {precision:.4f} | F1: {f1:.4f}\n")

if __name__ == "__main__":
    main()