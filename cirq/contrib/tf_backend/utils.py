import tensorflow as tf

def tensorboard_session(tensor, feed_dict, fout='./temp'):
    """Inspection tool for tensorboard."""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)
        print("outcome of arg")
        print(result)
        print("Graph protobuf")
        print([n.name for n in tf.get_default_graph().as_graph_def().node])


        # Merge all the summaries and write them out to fout
        merged = tf.summary.merge_all()
        temp_writer = tf.summary.FileWriter('./temp', sess.graph)
        tf.global_variables_initializer().run()

        summary = sess.run(merged, feed_dict=feed_dict)
        temp_writer.add_summary(summary, 0)
