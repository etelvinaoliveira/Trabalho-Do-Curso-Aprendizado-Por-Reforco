import argparse
import os
import numpy as np
import math
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from gym import Env, spaces
import time
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

def read_and_decode(dataset, batch_size, is_training, data_size, n_patients):
    """Prepara o dataset para treino ou validação."""
    if is_training:
        dataset = dataset.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.prefetch(buffer_size=data_size // batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat(None)
    return dataset

def initialize_clinical_practice(clinical_cases_feat, clinical_cases_labels, dataset_size, n_classes, is_training, n_patients, set_distribution):
    """Inicializa prática para treino ou validação, garantindo pesos não nulos e normalizados."""

    if is_training:
        _, counts = np.unique(clinical_cases_labels, return_counts=True)

        akiec = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 0), axis=0))
        akiec_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 0), axis=0))

        bcc = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 1), axis=0))
        bcc_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 1), axis=0))

        bkl = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 2), axis=0))
        bkl_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 2), axis=0))

        df = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 3), axis=0))
        df_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 3), axis=0))

        mel = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 4), axis=0))
        mel_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 4), axis=0))

        nv = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 5), axis=0))
        nv_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 5), axis=0))

        vasc = np.squeeze(np.take(clinical_cases_feat, np.where(clinical_cases_labels == 6), axis=0))
        vasc_labels = np.squeeze(np.take(clinical_cases_labels, np.where(clinical_cases_labels == 6), axis=0))

        set_distribution = np.maximum(set_distribution, 1e-5)  # evita zeros
        set_distribution = set_distribution / np.sum(set_distribution)  # normaliza novamente
        set_distribution = set_distribution.astype(np.float32)

        akiec_set = tf.data.Dataset.from_tensor_slices((akiec, akiec_labels)).shuffle(buffer_size=counts[0], reshuffle_each_iteration=True).repeat()
        bcc_set = tf.data.Dataset.from_tensor_slices((bcc, bcc_labels)).shuffle(buffer_size=counts[1], reshuffle_each_iteration=True).repeat()
        bkl_set = tf.data.Dataset.from_tensor_slices((bkl, bkl_labels)).shuffle(buffer_size=counts[2], reshuffle_each_iteration=True).repeat()
        df_set = tf.data.Dataset.from_tensor_slices((df, df_labels)).shuffle(buffer_size=counts[3], reshuffle_each_iteration=True).repeat()
        mel_set = tf.data.Dataset.from_tensor_slices((mel, mel_labels)).shuffle(buffer_size=counts[4], reshuffle_each_iteration=True).repeat()
        nv_set = tf.data.Dataset.from_tensor_slices((nv, nv_labels)).shuffle(buffer_size=counts[5], reshuffle_each_iteration=True).repeat()
        vasc_set = tf.data.Dataset.from_tensor_slices((vasc, vasc_labels)).shuffle(buffer_size=counts[6], reshuffle_each_iteration=True).repeat()

        dataset_train = tf.data.Dataset.sample_from_datasets(
            [akiec_set, bcc_set, bkl_set, df_set, mel_set, nv_set, vasc_set],
            weights=set_distribution
        ).batch(1)

    else:
        dataset_train = tf.data.Dataset.from_tensor_slices((clinical_cases_feat, clinical_cases_labels))
        dataset_train = read_and_decode(dataset_train, 1, is_training, dataset_size, n_patients)

    patients = iter(dataset_train)
    return patients

def get_next_patient(patients):
    """Recupera próximo paciente."""
    patient_scores, patient_diagnostics = patients.get_next()
    return np.squeeze(patient_scores), patient_diagnostics.numpy()[0]

class Dermatologist(Env):
    def __init__(self, patients, n_classes, vocab):
        self.action_space = spaces.Discrete(len(vocab))
        self.observation_space = spaces.Box(-1 * math.inf * np.ones((n_classes,)), math.inf * np.ones((n_classes,)))
        n_state, n_gt = get_next_patient(patients)
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt
        self.number_of_patients = 0

    def step(self, patients, n_patients, vocab, action):
        """Realiza uma ação e atualiza o estado."""
        reward_table = np.array([[2, -2, -3, -3, -2, -3, -3, -1],
                                 [-2, 3, -4, -4, -2, -4, -4, -1],
                                 [-2, -2, 1, -2, -3, -2, -2, -1],
                                 [-2, -2, -2, 1, -3, -2, -2, -1],
                                 [-4, -3, -5, -5, 5, -5, -5, -1],
                                 [-2, -2, -2, -2, -3, 1, -2, -1],
                                 [-2, -2, -2, -2, -3, -2, 1, -1]], np.float32)

        reward = reward_table[self.gt, action]
        revised_state = tf.one_hot(action, len(vocab))
        n_state, n_gt = get_next_patient(patients)

        old_gt = self.gt
        self.state = n_state
        self.gt = n_gt
        self.number_of_patients += 1
        done = 1 if self.number_of_patients >= n_patients else 0

        return revised_state, self.state, reward, done, old_gt

    def reset(self, clinical_cases_feat, clinical_cases_labels, n_classes, dataset_size, vocab, is_training, n_patients, sample_distribution):
        """Reseta o ambiente para novo episódio."""
        patients = initialize_clinical_practice(clinical_cases_feat, clinical_cases_labels, dataset_size,
                                                  n_classes, is_training, n_patients, sample_distribution)
        n_state, n_gt = get_next_patient(patients)
        self.state = n_state
        self.revised_state = self.state
        self.gt = n_gt
        self.number_of_patients = 0
        return self.state, patients


def main(_):
    gamma = 0.7           # Fator de desconto
    epsilon = 0.1           # Taxa de exploração inicial
    learning_rate = 0.1   
    tf1.enable_eager_execution()

    database = pd.read_csv('data/vectorDB.csv')
    labels = np.asarray(database['dx'])
    labels[labels == 'scc'] = 'akiec'

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    vocab = le.classes_
    n_words = len(vocab)

    if Flags.use_unknown:
        vocab = np.append(vocab, 'unkn')

    features1 = np.load("data/nmed_rn34_ham10k_vectors.npy")
    features2 = pd.read_csv("data/vectorDB.csv")
    features2.pop('dx')
    features2 = np.asarray(features2, dtype='float32')
    features = np.concatenate([features1, features2], axis=1)

    _, counts = np.unique(labels, return_counts=True)
    counts = counts / np.sum(counts)

    labels_cat = le.transform(labels)

    train_feat, val_feat, train_labels, val_labels = train_test_split(features, labels_cat, test_size=0.2,
                                                                       random_state=111, stratify=labels_cat)
    

    patients = initialize_clinical_practice(train_feat, train_labels, train_labels.shape[0], True,
                                            n_words, Flags.n_patients, counts)

    derm = Dermatologist(patients, n_words, vocab)

    state_dim = derm.state.shape[0]
    num_actions = Flags.n_actions
    weights = np.random.randn(state_dim, num_actions) * 0.01


    episode_reward_history = []
    episode_val_reward_history = []
    validation_bacc_history = []
    mel_history = []
    unk_history = []
    best_bacc = 0
    best_reward = -math.inf
    history_report_bacc = None
    history_cov_bacc = None
    history_report_reward = None
    history_cov_reward = None
    best_bacc = 0
    best_reward = -1 * math.inf

    for episode in range(Flags.n_episodes):
        print('Starting episode ',episode)
        state, patients = derm.reset(train_feat, train_labels, train_labels.shape[0], n_words, vocab, True, Flags.n_patients, counts)
        done = False
        episode_score = 0

        while not done:
           
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                q_values = weights.T.dot(state)
                action = np.argmax(q_values)

   
            _, next_state, reward, done, _ = derm.step(patients, Flags.n_patients, vocab, action)

            episode_score += reward


            next_q_values = weights.T.dot(next_state)
            next_action = np.argmax(next_q_values)

            target = reward + gamma * next_q_values[next_action]
            current_q = weights.T.dot(state)[action]
            delta = target - current_q
            weights[:, action] += learning_rate * delta * state

            state = next_state

        print('The episode duration was', episode_score) 
        print('The episode reward was', episode_score)

        episode_reward_history.append(episode_score)

        state, patients_val = derm.reset(val_feat, val_labels, val_labels.shape[0], n_words, vocab, False, Flags.n_patients, counts)

        done = False
        error = []
        true_label = []
        mel_count = 0
        unk_count = 0
        episode_val_score = 0

        while not done:
            true_label.append(derm.gt)

            q_values = weights.T.dot(state)
            action = np.argmax(q_values)

            error.append(action)

            if vocab[action] == 'mel':
                mel_count += 1
            elif vocab[action] == 'unkn':
                unk_count += 1

            _, state, reward, done, _ = derm.step(patients_val, val_labels.shape[0], vocab, action)

            episode_val_score += reward

        bacc = metrics.balanced_accuracy_score(true_label, error)

        print('The reward of the validation episode was', episode_val_score)
        print('The balanced accuracy was', bacc)
        print('\n')

        episode_val_reward_history.append(episode_val_score)
        validation_bacc_history.append(bacc)
        mel_history.append(mel_count)
        unk_history.append(unk_count)

        if best_bacc < bacc:
            history_report_bacc = metrics.classification_report(true_label, error, digits=3, zero_division=0)
            history_cov_bacc = metrics.confusion_matrix(true_label, error)
            best_bacc = bacc

        if best_reward < episode_val_score:
            history_cov_reward = metrics.confusion_matrix(true_label, error)
            history_report_reward = metrics.classification_report(true_label, error, digits=3)
            best_reward = episode_val_score

        ## Retorna para treino
        _, patients = derm.reset(train_feat, train_labels, train_labels.shape[0], n_words, vocab, True, Flags.n_patients, counts)

    # ===========================================================
    # Finalização e Resultado
    # ===========================================================
    print('\nThe scores for best validation BAcc are:')
    print(history_report_bacc)
    print(history_cov_bacc)
    print('The best BAcc was', best_bacc)

    print('\nThe scores for best validation Reward are:')
    print(history_report_reward)
    print(history_cov_reward)
    print('The best reward was', best_reward)

    # ===========================================================
    # Plotagem dos Resultados
    # ===========================================================
    plt.figure(1)
    plt.plot(episode_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Train')
    plt.show()

    plt.figure(2)
    plt.plot(episode_val_reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Per Episode - Val')
    plt.show()

    plt.figure(3)
    plt.plot(validation_bacc_history)
    plt.xlabel('Episodes')
    plt.ylabel('Validation Balanced Accuracy')
    plt.show()

    plt.figure(4)
    plt.plot(mel_history)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Melanoma Decisions')
    plt.show()

    plt.figure(5)
    plt.plot(unk_history)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Unknown Decisions')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_patients', type=int, default=100, help='Número de pacientes por episódio')
    parser.add_argument('--n_episodes', type=int, default=150, help='Número de episódios para treino')
    parser.add_argument('--use_unknown', type=bool, default=False, help='Usa ação "unknown" ou não')
    parser.add_argument('--n_actions', type=int, default=7, help='Número de ações possíveis')
    Flags, unparsed = parser.parse_known_args()
    tf1.app.run(main=main)