import tensorflow as tf
import wordEmbedding as w2v


# ----------- GLOBAL VARIABLES ------------ #

# Shared Global Variables
BATCH_SIZE = 10
ENCODER_MAX_TIME = 20							# max length of input signal (in vectorised form)
DECODER_MAX_TIME = 36							# max scentence length (in wordvectors)

# Unique Global Variables
ENCODER_INPUT_DEPTH = 20						# size of a sound feature vector
DECODER_INPUT_DEPTH = 50  						# len(a_word_vector)
OUTPUT_VOCAB_SIZE = w2v.vocabSize 				# how many words in your vocab
encoder_hidden_units = 100						# arbitary choice atm
decoder_hidden_units = 100
LOSS_TYPE = 'index'								# choose from: 'index', 'cosine', 'euclid'



# ----------- MODEL FUNCTIONS ------------- #

tf.reset_default_graph()

def io():		# The input to our model (encoder) is the pre-embedding sound vector. The input into the decoder is the time shifted word embeddings of the text
	encoder_inputs_embedded = tf.placeholder(shape=(BATCH_SIZE, ENCODER_MAX_TIME, ENCODER_INPUT_DEPTH), dtype=tf.float32, name='encoder_inputs')  # [batch_size*length_of_sequence*20]
	decoder_targets_indicies = tf.placeholder(shape=(BATCH_SIZE, DECODER_MAX_TIME), dtype=tf.int32, name='decoder_targets')	# [batch_size, max_time36]
	decoder_inputs_embedded = tf.placeholder(shape=(BATCH_SIZE, DECODER_MAX_TIME, DECODER_INPUT_DEPTH), dtype=tf.float32, name='decoder_inputs')
	embed_normed = tf.constant(w2v.wordVectorsNormalised, dtype=tf.float32)

	# IMPORTANT - Decoder targets is the sequence we want the network to output (scentence + EOS). During training,
	# decoder inputs is the time lagged version of this sequence (EOS + scentence). During evaluation, decoder inputs
	# is the predicted word from the previous time point.

	return encoder_inputs_embedded, decoder_inputs_embedded, decoder_targets_indicies, embed_normed


def inference(encoder_inputs_embedded, decoder_inputs_embedded):

	#encoder_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([single_lstm_cell] * numLayersLSTM)

	# encoder
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
	_, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=False)  # replace later with bidirectional dynamic rnn

	# decoder
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)
	decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, initial_state=encoder_final_state, dtype=tf.float32, time_major=False, scope="plain_decoder")  # replace later with bidirectional dynamic rnn

	# output
	if LOSS_TYPE == 'index':
		decoder_logits = tf.contrib.layers.linear(decoder_outputs, OUTPUT_VOCAB_SIZE)										# (BATCH_SIZE, DECODER_MAX_TIME, OUTPUT_VOCAB_SIZE)
	else:
		decoder_logits = tf.contrib.layers.linear(decoder_outputs, DECODER_INPUT_DEPTH)  # (BATCH_SIZE, DECODER_MAX_TIME, OUTPUT_VOCAB_SIZE)

	return decoder_outputs, decoder_logits


def indexLoss(decoder_targets_inds, decoder_logits):

	with tf.name_scope('performance_metrics'):
		# loss
		one_hot = tf.one_hot(decoder_targets_inds, depth=OUTPUT_VOCAB_SIZE,
							 dtype=tf.float32)  # decoder_targets (BATCH_SIZE, DECODER_MAX_TIME) containing the indicies of the correct words
		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=decoder_logits)

		decoder_prediction = tf.argmax(decoder_logits, 2)

		total_loss = tf.reduce_mean(stepwise_cross_entropy)
		tf.summary.scalar('summaries/total_loss', total_loss)

		with tf.name_scope('correct_prediction'):
			amax = tf.argmax(decoder_logits, axis=2)
			correct_prediction = tf.equal(amax, tf.cast(decoder_targets_inds, tf.int64))
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)  # log model output

	return total_loss, decoder_prediction


def cosineLoss(decoder_targets_inds, decoder_logits, embed_normed): #, decoder_inputs_embedded):

	def cosineSimilarityArray(logits):																					# embed_normed is a list of the word vectors. logits is the output of the network (word vector prediction for every time point)
		shifted_cosine = tf.matmul(logits, tf.transpose(embed_normed)) + 1												# DECODER_MAX_TIME*DECODER_INPUT_DEPTH x DECODER_INPUT_DEPTH*OUTPUT_VOCAB_SIZE = DECODER_MAX_TIME*OUTPUT_VOCAB_SIZE
		return -1* shifted_cosine

	def getTargetCS(data):
		cs = []
		for i in range(DECODER_MAX_TIME):
			cs.append(data[1][i][data[0][i]])
		return tf.stack(cs)


	with tf.name_scope('performance_metrics'):

		decoder_logits_normalised = tf.nn.l2_normalize(decoder_logits, dim=2)											# normalise the network output	(lookup cosine similarity formula if unsure why)
		cosine_similarity = tf.map_fn(cosineSimilarityArray, decoder_logits_normalised)									# for the entire batch, compute the cosine similarity for every time point

		decoder_prediction = tf.argmax(cosine_similarity, 2)   															# (BATCH_SIZE, DECODER_MAX_TIME)	- pick maximum liklihood words for each position (the word index)

		#total_loss = tf.reduce_max(cosine_similarity, 2)																# compute the lo
		cos_sims = tf.map_fn(getTargetCS, (decoder_targets_inds, cosine_similarity), dtype=tf.float32)
		total_loss = tf.reduce_mean(cos_sims)
		#total_loss = tf.map_fn(getTargetCS, (decoder_logits_normalised, decoder_inputs_embedded), dtype=tf.float32)
		# cosine_similarity = tf.losses.cosine_distance(decoder_logits, decoder_targets_inds, dim=1)
	

		tf.summary.scalar('summaries/total_loss', total_loss)


		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(decoder_prediction, tf.cast(decoder_targets_inds, tf.int64))
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)  # log model output

	return total_loss, decoder_prediction


def euclidLoss(decoder_targets_inds, decoder_logits, embed_normed):  # , decoder_inputs_embedded):

	def cosineSimilarityArray(logits):  																				# embed_normed is a list of the word vectors. logits is the output of the network (word vector prediction for every time point)
		shifted_cosine = tf.matmul(logits, tf.transpose(embed_normed)) + 1  	# DECODER_MAX_TIME*DECODER_INPUT_DEPTH x DECODER_INPUT_DEPTH*OUTPUT_VOCAB_SIZE = DECODER_MAX_TIME*OUTPUT_VOCAB_SIZE
		return -1 * shifted_cosine

	def getTargetCS(data):
		cs = []
		for i in range(DECODER_MAX_TIME):
			cs.append(data[1][i][data[0][i]])
		return tf.stack(cs)

	with tf.name_scope('performance_metrics'):
		decoder_logits_normalised = tf.nn.l2_normalize(decoder_logits,
													   dim=2)  # normalise the network output	(lookup cosine similarity formula if unsure why)
		cosine_similarity = tf.map_fn(cosineSimilarityArray,
									  decoder_logits_normalised)  # for the entire batch, compute the cosine similarity for every time point

		decoder_prediction = tf.argmax(cosine_similarity, 2)  # (BATCH_SIZE, DECODER_MAX_TIME)	- pick maximum liklihood words for each position (the word index)

		# total_loss = tf.reduce_max(cosine_similarity, 2)																# compute the lo
		cos_sims = tf.map_fn(getTargetCS, (decoder_targets_inds, cosine_similarity), dtype=tf.float32)
		total_loss = tf.reduce_mean(cos_sims)
		# total_loss = tf.map_fn(getTargetCS, (decoder_logits_normalised, decoder_inputs_embedded), dtype=tf.float32)
		# cosine_similarity = tf.losses.cosine_distance(decoder_logits, decoder_targets_inds, dim=1)


		tf.summary.scalar('summaries/total_loss', total_loss)

		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(decoder_prediction, tf.cast(decoder_targets_inds, tf.int64))
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)  # log model output

	return total_loss, decoder_prediction


"""
def getTargetCS(data):

	cs = tf.tensordot(data[0][0], data[1][0], 1)
	print(cs)

	for i in range(1,DECODER_MAX_TIME):
		new = tf.tensordot(data[0][i], data[1][i], 1)
		print(new)
		cs = tf.concat([cs, new], axis=0)		# DECODER_MAX_TIME*DECODER_INPUT_DEPTH x DECODER_MAX_TIME*DECODER_INPUT_DEPTH = DECODER_MAX_TIME*OUTPUT_VOCAB_SIZE
		print(cs)

	print(cs)
	cs = tf.reduce_sum(cs)
	print(cs)
	return cs

"""

def optimise(total_loss):
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

	return train_step