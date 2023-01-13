import helper # Вспомогательные функции для сохранения и загрузки модели и параметров
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.contrib import seq2seq
from collections import Counter
# Все статьи
mode = "full"       

# Выборка (~800 статей)
# mode = "sample"  
data_dir = './data/headers_' + mode + '.txt'
text = helper.load_data(data_dir)
# Диапазон номеров выводимых заголовков
view_sentence_range = (0, 10)

print('Немного статистики')
print('Уникальных слов: {}'.format(len({word: None for word in text.split(" ")})))
sentences = [sentence for sentence in text.split('. ')]
print('Количество заголовков: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Среднее количество слов в заголовке: {}'.format(np.average(word_count_sentence)))

print()
print('Заголовки с {} по {}:'.format(*view_sentence_range))
print('\n'.join(text.split('. ')[view_sentence_range[0]:view_sentence_range[1]]))
def create_lookup_tables(text):

    counter = Counter(text)
    vocab = sorted(counter, key=counter.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    
    return vocab_to_int, int_to_vocab
def token_lookup():
    tokens = {
        ".": "||PERIOD||",
        ",": "||COMMA||",
        '"': "||QUOT_MARK||",
        ";": "||SEMICOL||",
        "!": "||EXCL_MARK||",
        "?": "||QUEST_MARK||",
        "(": "||L_PARENTH||",
        ")": "||R_PARENTH||",
        "--": "||DASH||",
        "\n": "||RETURN||"
    }
    
    return tokens
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables, text)
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
def get_inputs():
    inputs = tf.placeholder(tf.int32, shape=[None, None], name="input")
    targets = tf.placeholder(tf.int32, shape=[None, None], name="targets")
    l_rate = tf.placeholder(tf.float32, name="learning_rate")
    
    return inputs, targets, l_rate
def get_init_cell(batch_size, rnn_size, num_layers=2, dropout=0.1):
    basic_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    basic_cell_with_dropout = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=(1-dropout))
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_cell_with_dropout] * num_layers)
    init_state = tf.identity(multi_rnn_cell.zero_state(batch_size, tf.float32), name="initial_state")

    return multi_rnn_cell, init_state
def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    
    return tf.nn.embedding_lookup(embedding, input_data)
def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")
    
    return outputs, final_state
def build_nn(cell, rnn_size, input_data, vocab_size):
    
    inputs = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, inputs)
    logits = tf.contrib.layers.fully_connected(outputs, num_outputs=vocab_size, activation_fn=None)
    
    return logits, final_state
def get_batches(int_text, batch_size, seq_length):

    num_batches = len(int_text) // (batch_size * seq_length)
    batches = []
    
    for batch_idx in range(num_batches):
        inputs=[]
        targets=[]
        
        for seq_idx in range(batch_size):
            i = batch_idx * seq_length + seq_idx * seq_length
            inputs.append(int_text[i:i+seq_length])
            targets.append(int_text[i+1: i+seq_length+1])
        
        batches.append([inputs, targets])
    
    return np.array(batches)
def get_tensors(loaded_graph):

    inputs = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    
    return inputs, initial_state, final_state, probs
def pick_word(probabilities, int_to_vocab):

    return int_to_vocab[np.argmax(probabilities)]
first_words = list(set([line.split(" ")[0] for line in text.split('. ')]))
def infer(gen_length, prime_word, load_dir, rnn_layers, rnn_size):
    
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        
        # Загружаем модель
        loader = tf.train.import_meta_graph(load_dir +'_{}_{}.meta'.format(rnn_layers, rnn_size) )
        loader.restore(sess, load_dir +'_{}_{}'.format(rnn_layers, rnn_size))

        # Получаем тензоры из модели
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

        # Инициализируем переменную, где будем хранить сгенерированную последовательность
        gen_sentences = [prime_word]
    
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        # Генерация последовательности
        for n in range(gen_length):
            
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])
    
            # Получаем вероятности
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})
    
            # Получаем следующее слово
            pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)
            gen_sentences.append(pred_word)

        # Удаляем токены пунктуации, заменяя их на соответствующие символы
        headlines = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            headlines = headlines.replace(' ' + token.lower(), key)
        headlines = headlines.replace('\n ', '\n')
        headlines = headlines.replace('( ', '(')
        headlines = headlines.replace('. ', '.\n')
                
        print(headlines)
        return headlines
# Количество эпох (итераций обучения)
num_epochs = 20

# Размер батча
batch_size = 32

# Размер рекуррентного слоя
# rnn_size = 64

# Длина последовательности в батче, определяет как далеко "назад" сеть должна помнить контекст
seq_length = 10

# Скорость обучения
learning_rate = 0.01

# Выводить статистику через каждые N батчей
show_every_n_batches = 500

# Длина генерируемой последовательности
gen_length = 100

# массив опций для обучения; используется для того, чтобы определить наиболее удачную конфигурацию нейронной сети; 
# для обучения только по одному набору параметров, можно закомментировать остальные наборы
options = [
#     {
#         'rnn_size': 32,
#         'rnn_layers': 1
#     },
    {
        'rnn_size': 64,
        'rnn_layers': 1
    },
#     {
#         'rnn_size': 32,
#         'rnn_layers': 2
#     },
#     {
#         'rnn_size': 64,
#         'rnn_layers': 2
#     }
]

# Первая часть названия файла модели
save_dir = './models/word_emb_'
prime_word = first_words[random.randint(0, len(first_words))]

for option in options:

    option['start'] = time.time()
    print('Обучение модели Word Embeddings с параметрами (слои: {}, размер: {})'.format(option['rnn_layers'], option['rnn_size']))
    
    with open('results_word_emb_{}_{}.txt'.format(option['rnn_layers'], option['rnn_size']), 'a') as file:
        train_graph = tf.Graph()
        
        # Строим граф вычислений
        with train_graph.as_default():
            vocab_size = len(int_to_vocab)
            input_text, targets, lr = get_inputs()
            input_data_shape = tf.shape(input_text)
            cell, initial_state = get_init_cell(input_data_shape[0], option['rnn_size'], option['rnn_layers'])
            logits, final_state = build_nn(cell, option['rnn_size'], input_text, vocab_size)

            # Вычисляем вероятности для слов
            probs = tf.nn.softmax(logits, name='probs')

            # Функция потерь
            cost = seq2seq.sequence_loss(
                logits,
                targets,
                tf.ones([input_data_shape[0], input_data_shape[1]]))

            # Функция оптимизация
            optimizer = tf.train.AdamOptimizer(lr)
            # На мой взгляд, Adam выдает чуть более качественные результаты, чем RMSProp,
            # но вы можете попробовать и его:
            # optimizer = tf.train.RMSPropOptimizer(lr)

            # Вычисляем градиенты
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            train_op = optimizer.apply_gradients(capped_gradients)
        
        # Создаем батчи
        batches = get_batches(int_text, batch_size, seq_length)

        # Запускаем обучение
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            # Итерация по эпохам
            for epoch_i in range(num_epochs):
                state = sess.run(initial_state, {input_text: batches[0][0]})
                
                # Проходимся по всем батчам
                for batch_i, (x, y) in enumerate(batches):
                    
                    # Задаем входные и целевые данные, состояние и скорость обученя
                    feed = {
                        input_text: x,
                        targets: y,
                        initial_state: state,
                        lr: learning_rate}
                    
                    # Отправляем все это добро на обучение
                    train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                    # Для красоты и удобства считаем, сколько еще времени осталось на обучение
                    t = time.time()
                    time_diff = t - option['start']
                    rem_batches = num_epochs * len(batches) - (epoch_i * len(batches) + batch_i)
                    total_time = round((num_epochs * len(batches) / (epoch_i * len(batches) + batch_i + 1)) * time_diff)
                    rem_time = round(total_time - time_diff)
                    m, s = divmod(rem_time, 60)
                    h, m = divmod(m, 60)
                    
                    # Выводим статистику каждые  батчей
                    if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                        print(' ' * 100, end='\r')
                        print('Итерация {:>3} Батч {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))
                    
                    print("Оценка оставшегося времени на обучение по текущему набору параметров: %d:%02d:%02d " % (h, m, s), end="\r")
                
                # Сохраняем модель
                saver = tf.train.Saver()
                saver.save(sess, save_dir + "_{}_{}".format(option['rnn_layers'], option['rnn_size']))
                print()
                
                # Смотрим, что получилось в этой итерации
                print("Сгенерированный текст для итерации {}:".format(epoch_i))
                generated_text = infer(gen_length, prime_word, save_dir, option['rnn_layers'], option['rnn_size'])

                # И также сохраняем в файл
                file.write('Итерация ' + str(epoch_i) + '\n')
                file.write(generated_text)

            # В конце считаем время, затраченное на обучение по текущему набору параметров
            time_diff = time.time() - option['start']
            m, s = divmod(time_diff, 60)
            h, m = divmod(m, 60)
            
            print("Затраченное время: %d:%02d:%02d " % (h, m, s))
            file.write("Total time for training this option: %d:%02d:%02d " % (h, m, s))
            print('Обучение модели Word Embeddings с параметрами {} {} завершено'.format(option['rnn_layers'], option['rnn_size']))