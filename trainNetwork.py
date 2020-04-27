import tensorflow as tf
import datetime
import seq2seqModel as model
import inputOps as data_input


# ----------- GLOBAL VARIABLES ------------ #

# Shared Global Variables
BATCH_SIZE = 10
ENCODER_MAX_TIME = 20 			#arbitary but less than shortest input vector of sound
DECODER_MAX_TIME = 36

# Unique Global variables
NUM_INTERATIONS = 500000



# ----------- SEQ2SEQ MODEL -------------- #

# input
encoder_inputs_embedded, decoder_inputs_embedded, decoder_targets_indicies, embed_normed = model.io()				# using the feed dictionary
# seq2seq model
decoder_outputs, decoder_logits = model.inference(encoder_inputs_embedded, decoder_inputs_embedded)
# loss
if model.LOSS_TYPE == 'index':
	loss, decoder_prediction = model.indexLoss(decoder_targets_indicies, decoder_logits)
elif model.LOSS_TYPE == 'cosine':
	loss, decoder_prediction = model.cosineLoss(decoder_targets_indicies, decoder_logits, embed_normed)
elif model.LOSS_TYPE == 'euclid':
	loss, decoder_prediction = model.euclidLoss(decoder_targets_indicies, decoder_logits, embed_normed)

# training operation
train_step = model.optimise(loss)



# ----------- initialisations ------------ #
init_op = tf.group(tf.tables_initializer(), tf.global_variables_initializer(),tf.local_variables_initializer())  		# Create the graph, etc.
saver = tf.train.Saver()

with tf.Session() as sess:

	sess.run(init_op)

	# Tensorboard - merge all the summaries and write them out to directory
	merged = tf.summary.merge_all()
	LOG_DIR = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	writer = tf.summary.FileWriter(LOG_DIR, sess.graph)


# ----------- TRAINING LOOP -------------- #
	
	for i in range(NUM_INTERATIONS):

		encoderTrain, _, decoderTargetsIndicies, decoderInputTrain = data_input.getTrainingBatch(BATCH_SIZE)			# decoder inputs are lagged
		feedDict = {encoder_inputs_embedded: encoderTrain, decoder_targets_indicies: decoderTargetsIndicies, decoder_inputs_embedded: decoderInputTrain}

		try:
			curLoss, _ = sess.run([loss, train_step], feed_dict=feedDict)
		except ValueError:
			print('EEERRRROOORRRRR!')


		if i % 50 == 0:
			print('Current loss:', curLoss, 'at iteration', i)
			encoderTrain, _, decoderTargetsIndicies, decoderInputTrain = data_input.getTestBatch(BATCH_SIZE)
			feedDict = {encoder_inputs_embedded: encoderTrain, decoder_targets_indicies: decoderTargetsIndicies, decoder_inputs_embedded: decoderInputTrain}
			
			summary, pred = sess.run([merged, decoder_prediction], feed_dict=feedDict)
			writer.add_summary(summary, global_step=i)

			for j in range(BATCH_SIZE):
				print(data_input.w2v.idsToSentence(pred[j]))


		if i % 10000 == 0 and i != 0:
			print('Saving Checkpoint...')
			savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)


	"""
# ------------- EVALUATION --------------- #
	print('restoring model')
	saver.restore(sess, "trained_models/pretrained_seq2seq.ckpt-490000")
	print("Model restored.")

	encoderTrain, decoderTargetTrain, decoderInputTrain, label_inds = data_input.getTestBatch(BATCH_SIZE)
	feedDict = {encoder_inputs_embedded: encoderTrain}

	pred = sess.run(decoder_prediction, feedDict)
	print(pred)
	"""