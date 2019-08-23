import numbers
import tensorflow as tf


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]


def get_at_indices(tensor, indices):
    """ Like `tensor[np.arange(len(tensor)), indices]` in numpy. """
    counter = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
    return tf.gather_nd(tensor, tf.stack((counter, indices), -1))


def batch_hard_triplet(dists, pids, margin):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("batch_hard_triplet"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                    (dists, negative_mask), tf.float32)
        # Another way of achieving the same, though more hacky:
        # closest_negative = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative
        if isinstance(margin, numbers.Real):
            diff = tf.maximum(diff + margin, 0.0)
        elif margin == 'soft':
            diff = tf.nn.softplus(diff)
        elif margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(margin))

    return diff

    # if batch_precision_at_k is None:
    #     return diff
    #
    # # For monitoring, compute the within-batch top-1 accuracy and the
    # # within-batch precision-at-k, which is somewhat more expressive.
    # with tf.name_scope("monitoring"):
    #     # This is like argsort along the last axis. Add one to K as we'll
    #     # drop the diagonal.
    #     _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)  # 查找距离最近的k个点 nn.top_k along the last dimension
    #
    #     # Drop the diagonal (distance to self is always least).
    #     indices = indices[:,1:]  # 消除第一个点 即：它自己本身 indices.shape = [batch_size, k-1]
    #
    #     # Generate the index indexing into the batch dimension.
    #     # This is something like [[0,0,0],[1,1,1],...,[B,B,B]]
    #     batch_index = tf.tile(
    #         tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
    #         (1, tf.shape(indices)[1]))
    #
    #     # Stitch the above together with the argsort indices to get the
    #     # indices of the top-k of each row.
    #     topk_indices = tf.stack((batch_index, indices), -1)
    #     # topk_indices.shape = [batch_size,3,2]
    #
    #     # See if the topk belong to the same person as they should, or not.
    #     topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)
    #     # tf.gather_nd: Gather slices from 'mask' into a Tensor with shape specified by 'indices'.
    #     # 相当于 0 号 取【0，a】 【0，b】【0，c】 1 号 取【1，d】【1，e】【1，f】
    #     # 若取对了，则+1 若取错了，则+0
    #
    #
    #
    #     # All of the above could be reduced to the simpler following if k==1
    #     #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])
    #
    #     topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
    #     top1 = tf.reduce_mean(topk_is_same_f32[:,0])
    #     prec_at_k = tf.reduce_mean(topk_is_same_f32)
    #     # top1 取对的 mean 和 所有top k 取对 的 mean
    #     # 取对的地方值为1 不然为0 故越大越好
    #
    #     # Finally, let's get some more info that can help in debugging while
    #     # we're at it!
    #     # 当 boolean_mask 的维度一致时， 输出1维矩阵
    #     negative_dists = tf.boolean_mask(dists, negative_mask)
    #     positive_dists = tf.boolean_mask(dists, positive_mask)
    #
    #     return diff, top1, prec_at_k, topk_is_same, negative_dists, positive_dists


def triplet_loss(endpoints, pids, margin,model_type, batch_precision_at_k=None, metric='euclidean'):

    with tf.name_scope('triplet_loss'):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))
        dists = cdist(endpoints['emb'], endpoints['emb'], metric=metric)
        losses= batch_hard_triplet(
        dists, pids, margin)

        if batch_precision_at_k is None:
            return losses

        # For monitoring, compute the within-batch top-1 accuracy and the
        # within-batch precision-at-k, which is somewhat more expressive.
        with tf.name_scope("triplet_monitoring"):
            # This is like argsort along the last axis. Add one to K as we'll
            # drop the diagonal.
            _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)  # 查找距离最近的k个点 nn.top_k along the last dimension

            # Drop the diagonal (distance to self is always least).
            indices = indices[:,1:]  # 消除第一个点 即：它自己本身 indices.shape = [batch_size, k-1]

            # Generate the index indexing into the batch dimension.
            # This is something like [[0,0,0],[1,1,1],...,[B,B,B]]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
                (1, tf.shape(indices)[1]))

            # Stitch the above together with the argsort indices to get the
            # indices of the top-k of each row.
            topk_indices = tf.stack((batch_index, indices), -1)
            # topk_indices.shape = [batch_size,3,2]

            # See if the topk belong to the same person as they should, or not.
            topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)
            # tf.gather_nd: Gather slices from 'mask' into a Tensor with shape specified by 'indices'.
            # 相当于 0 号 取【0，a】 【0，b】【0，c】 1 号 取【1，d】【1，e】【1，f】
            # 若取对了，则+1 若取错了，则+0

            # All of the above could be reduced to the simpler following if k==1
            #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])

            topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
            top1 = tf.reduce_mean(topk_is_same_f32[:,0])
            prec_at_k = tf.reduce_mean(topk_is_same_f32)
            # top1 取对的 mean 和 所有top k 取对 的 mean
            # 取对的地方值为1 不然为0 故越大越好

            # Finally, let's get some more info that can help in debugging while
            # we're at it!
            # 当 boolean_mask 的维度一致时， 输出1维矩阵
            negative_dists = tf.boolean_mask(dists, negative_mask)
            positive_dists = tf.boolean_mask(dists, positive_mask)

    return losses, top1, prec_at_k, topk_is_same, negative_dists, positive_dists




def merged_loss(endpoints, pids, margin,model_type,batch_precision_at_k=None, metric='euclidean'):

    with tf.name_scope('merged_loss'):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))
        dists = cdist(endpoints['emb'], endpoints['emb'], metric=metric)

        if model_type == 'resnet_v1_50':
            dists1 = cdist(endpoints['feature1'], endpoints['feature1'], metric=metric)
            losses1 = batch_hard_triplet(
                dists1, pids, margin)

            dists2 = cdist(endpoints['feature2'], endpoints['feature2'], metric=metric)
            losses2= batch_hard_triplet(
                dists2, pids, margin)

            dists3 = cdist(endpoints['feature3'], endpoints['feature3'], metric=metric)
            losses3= batch_hard_triplet(
                dists3, pids, margin)

            dists4 = cdist(endpoints['feature4'], endpoints['feature4'], metric=metric)
            losses4= batch_hard_triplet(
                dists4, pids, margin)

            dists_fu = cdist(endpoints['fusion_layer'], endpoints['fusion_layer'], metric=metric)
            losses_fu  = batch_hard_triplet(
                dists_fu, pids, margin)

            losses = losses1 + losses2 + losses3 + losses4 + losses_fu

        elif model_type == 'mobilenet_v1_1_224':
            dists1 = cdist(endpoints['feature1'], endpoints['feature1'], metric=metric)
            losses1= batch_hard_triplet(
                dists1, pids, margin)
            dists2 = cdist(endpoints['feature2'], endpoints['feature2'], metric=metric)
            losses2 = batch_hard_triplet(
                dists2, pids,margin)
            dists3 = cdist(endpoints['feature3'], endpoints['feature3'], metric=metric)
            losses3 = batch_hard_triplet(
                dists3, pids, margin)
            dists4 = cdist(endpoints['feature4'], endpoints['feature4'], metric=metric)
            losses4 = batch_hard_triplet(
                dists4, pids, margin)
            dists5 = cdist(endpoints['feature5'], endpoints['feature5'], metric=metric)
            losses5 = batch_hard_triplet(
                dists5, pids, margin)
            dists_fu = cdist(endpoints['fusion_layer'], endpoints['fusion_layer'], metric=metric)
            losses_fu = batch_hard_triplet(
                dists_fu, pids, margin)

            losses = losses1 + losses2 + losses3 + losses4 + losses5 + losses_fu

        else :
            print('no such model ,choose from resnet_v1_50 or mobilenet_v1_1_224')
            exit(1)

        if batch_precision_at_k is None:
            return losses

        # For monitoring, compute the within-batch top-1 accuracy and the
        # within-batch precision-at-k, which is somewhat more expressive.
        with tf.name_scope("monitoring"):
            # This is like argsort along the last axis. Add one to K as we'll
            # drop the diagonal.
            _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)  # 查找距离最近的k个点 nn.top_k along the last dimension

            # Drop the diagonal (distance to self is always least).
            indices = indices[:,1:]  # 消除第一个点 即：它自己本身 indices.shape = [batch_size, k-1]

            # Generate the index indexing into the batch dimension.
            # This is something like [[0,0,0],[1,1,1],...,[B,B,B]]
            batch_index = tf.tile(
                tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
                (1, tf.shape(indices)[1]))

            # Stitch the above together with the argsort indices to get the
            # indices of the top-k of each row.
            topk_indices = tf.stack((batch_index, indices), -1)
            # topk_indices.shape = [batch_size,3,2]

            # See if the topk belong to the same person as they should, or not.
            topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)
            # tf.gather_nd: Gather slices from 'mask' into a Tensor with shape specified by 'indices'.
            # 相当于 0 号 取【0，a】 【0，b】【0，c】 1 号 取【1，d】【1，e】【1，f】
            # 若取对了，则+1 若取错了，则+0

            # All of the above could be reduced to the simpler following if k==1
            #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])

            topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
            top1 = tf.reduce_mean(topk_is_same_f32[:,0])
            prec_at_k = tf.reduce_mean(topk_is_same_f32)
            # top1 取对的 mean 和 所有top k 取对 的 mean
            # 取对的地方值为1 不然为0 故越大越好

            # Finally, let's get some more info that can help in debugging while
            # we're at it!
            # 当 boolean_mask 的维度一致时， 输出1维矩阵
            negative_dists = tf.boolean_mask(dists, negative_mask)
            positive_dists = tf.boolean_mask(dists, positive_mask)

    return losses, top1, prec_at_k, topk_is_same, negative_dists, positive_dists





LOSS_CHOICES = {
    'triplet_loss': triplet_loss,
    'merged_loss': merged_loss
}
