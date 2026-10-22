import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.build_metadata import EXPECTED_VARIANTS


# ==========================================================
# COPPIE DI ATTRIBUTI DA VALUTARE PER LO SPLIT OOD
# ==========================================================
#
# Ogni coppia viene analizzata per verificare:
# - quante combinazioni di valori sono effettivamente presenti;
# - quanti esempi appartengono a ciascuna combinazione;
# - se esiste una combinazione sufficientemente rappresentativa
#   da poter essere trattenuta per una valutazione OOD composizionale.
#
# La scelta definitiva della coppia e della combinazione OOD
# verrà effettuata nel blocco successivo.
CANDIDATE_PAIRS = [
    ("hair_color", "face_shape"),
    ("eye_color", "chin_length"),
    ("hair_color", "chin_length"),
    ("eye_color", "face_shape"),
    ("glasses", "face_shape"),
]


def build_attribute_counts(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    # Costruisce le distribuzioni marginali di tutti gli attributi.

    # Il risultato contiene una riga per ogni coppia
    # attributo-valore osservata, con conteggio assoluto e frequenza
    # relativa rispetto al numero totale di esempi.

    # Ogni tabella prodotta nel ciclo descrive un singolo attributo;
    # alla fine le uniamo in un unico DataFrame long-format.
    tables = []

    # EXPECTED_VARIANTS definisce la lista canonica dei 18 attributi.
    # Iterare su questa struttura mantiene l'audit coerente con
    # la validazione usata durante la costruzione dei metadata.
    for attribute in EXPECTED_VARIANTS:
        # Conta quante immagini assumono ciascun valore dell'attributo.
        #
        # sort_index() ordina i valori discreti in senso numerico,
        # così 0, 1, 2, ... compaiono nel loro ordine naturale.
        counts = (
            metadata[attribute]
            .value_counts()
            .sort_index()
            .rename_axis("value")
            .reset_index(name="count")
        )

        # Aggiungiamo il nome dell'attributo come prima colonna,
        # così tutte le distribuzioni possono convivere nello stesso CSV.
        counts.insert(0, "attribute", attribute)

        # La frequenza relativa consente il confronto fra attributi
        # anche se in futuro l'audit verrà applicato a dataset di dimensioni diverse.
        counts["frequency"] = counts["count"] / len(metadata)

        tables.append(counts)

    # ignore_index=True produce un indice consecutivo e non conserva
    # gli indici locali generati per ogni singolo attributo.
    return pd.concat(tables, ignore_index=True)


def build_pair_counts(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    # Costruisce le distribuzioni congiunte delle coppie candidate OOD.

    # Per ogni coppia vengono restituiti i valori dei due attributi,
    # il numero di esempi che condividono quella combinazione
    # e la frequenza relativa sul dataset.

    # Ogni elemento di tables conterrà le combinazioni osservate
    # per una delle coppie definite in CANDIDATE_PAIRS.
    tables = []

    for first, second in CANDIDATE_PAIRS:
        # groupby conta il numero di righe per ogni combinazione reale
        # dei due attributi, ad esempio hair_color=3 e face_shape=3.
        #
        # observed=True evita di produrre combinazioni non osservate
        # nel caso in cui le colonne vengano trattate come categoriche.
        counts = (
            metadata
            .groupby([first, second], observed=True)
            .size()
            .reset_index(name="count")
        )

        # Etichetta leggibile della coppia, utile per filtrare le righe
        # dopo la concatenazione di tutte le tabelle.
        counts.insert(0, "pair", f"{first}__{second}")

        # Uniformiamo i nomi delle colonne in tutte le tabelle:
        # ciò permette di confrontare coppie diverse nello stesso formato.
        #
        # Esempio:
        # hair_color e face_shape diventano first_value e second_value.
        counts = counts.rename(
            columns={
                first: "first_value",
                second: "second_value",
            }
        )

        # Frequenza relativa della singola combinazione sul totale dataset.
        counts["frequency"] = counts["count"] / len(metadata)

        tables.append(counts)

    return pd.concat(tables, ignore_index=True)


def main() -> None:
    # ======================================================
    # INTERFACCIA A RIGA DI COMANDO
    # ======================================================
    #
    # I valori di default consentono l'esecuzione standard:
    # python -m src.data.audit_metadata
    #
    # Gli argomenti permettono comunque di applicare lo stesso audit
    # a metadata o directory di output differenti.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/metadata.csv"),
        help="Percorso del metadata centralizzato in formato CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/audit"),
        help="Directory in cui salvare gli artefatti dell'audit.",
    )

    args = parser.parse_args()

    # La cartella viene creata in modo idempotente:
    # nessun errore se esiste già da un'esecuzione precedente.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Carichiamo il metadata una sola volta e lo riutilizziamo
    # per tutte le elaborazioni seguenti.
    metadata = pd.read_csv(args.metadata)

    # Produciamo i due artefatti tabellari dell'audit:
    # distribuzioni marginali e distribuzioni congiunte.
    attribute_counts = build_attribute_counts(metadata)
    pair_counts = build_pair_counts(metadata)

    # CSV long-format delle frequenze per valore di attributo.
    # È utile sia per visualizzazioni sia per verifiche automatiche.
    attribute_counts.to_csv(
        args.output_dir / "attribute_counts.csv",
        index=False,
    )

    # CSV delle frequenze per combinazione delle coppie candidate OOD.
    pair_counts.to_csv(
        args.output_dir / "candidate_pair_counts.csv",
        index=False,
    )

    # ======================================================
    # RIEPILOGO JSON
    # ======================================================
    #
    # Il JSON conserva statistiche aggregate leggere, facilmente
    # ispezionabili senza dover rileggere l'intero CSV metadata.
    summary = {
        "num_examples": int(len(metadata)),
        "attributes": {},
        "candidate_pairs": {},
    }

    # Per ciascun attributo salviamo cardinalità attesa e osservata,
    # più la frequenza minima e massima tra i suoi valori.
    for attribute, num_variants in EXPECTED_VARIANTS.items():
        counts = metadata[attribute].value_counts()

        summary["attributes"][attribute] = {
            # Numero di categorie dichiarate dalla specifica del dataset.
            "expected_num_variants": num_variants,

            # Numero di categorie effettivamente presenti nel metadata.
            "observed_num_variants": int(counts.size),

            # Un minimo basso può segnalare una categoria rara,
            # utile da conoscere prima di definire split stratificati.
            "minimum_count": int(counts.min()),

            # Un massimo alto, confrontato con il minimo, aiuta a valutare
            # l'eventuale sbilanciamento della distribuzione.
            "maximum_count": int(counts.max()),
        }

    # Per ogni coppia candidata calcoliamo statistiche sulle sole
    # combinazioni realmente osservate nel dataset.
    for first, second in CANDIDATE_PAIRS:
        pair_name = f"{first}__{second}"

        # Filtriamo la tabella cumulativa per ottenere solo le righe
        # associate alla coppia che stiamo riassumendo.
        selected = pair_counts[
            pair_counts["pair"] == pair_name
        ]

        summary["candidate_pairs"][pair_name] = {
            # Numero di combinazioni effettivamente presenti.
            "observed_combinations": int(len(selected)),

            # Conteggio della combinazione più rara.
            "minimum_count": int(selected["count"].min()),

            # Conteggio della combinazione più frequente.
            "maximum_count": int(selected["count"].max()),

            # Valore centrale: è utile per scegliere una combinazione OOD
            # rappresentativa, evitando valori estremamente rari o comuni.
            "median_count": float(selected["count"].median()),
        }

    # Salviamo un report JSON stabile e leggibile.
    summary_path = args.output_dir / "audit_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    # Stampiamo lo stesso riepilogo per avere un feedback immediato
    # quando lo script viene eseguito da notebook o terminale.
    print(json.dumps(summary, indent=2, sort_keys=True))


# Il blocco viene eseguito soltanto quando il file è invocato come modulo
# o script principale, non quando le funzioni vengono importate nei test.
if __name__ == "__main__":
    main()
