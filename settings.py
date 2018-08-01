from differential_privacy.dp_sgd.dp_optimizer import utils

class FlagSettings:
    def __init__(self, preserve_privacy=None):
        assert preserve_privacy is not None, "preserve_privacy must be True or False, not None."
        # data
        self.num_training_images = 60000
        self.num_testing_images = 10000
        self.image_size = 28

        # parameters for the training
        self.batch_size = 600 # The training batch size.
        self.batches_per_lot = 1 # Number of batches per lot.
        # Together, batch_size and batches_per_lot determine lot_size.
        self.num_training_steps = 100 # The number of training steps. This counts number of lots.

        self.randomize = True # If true, randomize the input data; otherwise use a fixed seed and non-randomized input.
        self.freeze_bottom_layers = False # If true, only train on the logit layer.
        self.save_mistakes = False # If true, save the mistakes made during testing.
        self.lr = 0.05 # start learning rate

        # For searching parameters
        self.projection_dimensions = 0 # PCA projection dimensions, or 0 for no projection.
        self.num_hidden_layers = 1 # Number of hidden layers in the network
        self.hidden_layer_num_units = 35 # Number of units per hidden layer
        self.default_gradient_l2norm_bound = 3.0 # norm clipping
        self.num_conv_layers = 0 # Number of convolutional layers to use.

        self.data_dir = "/tmp/mnist/"
        self.training_data_path =  self.data_dir + "mnist_train.tfrecord" # Location of the training data.
        self.eval_data_path =  self.data_dir + "mnist_test.tfrecord" # Location of the eval data.
        self.eval_steps = 10 # Evaluate the model every eval_steps

        # Parameters for privacy spending. We allow linearly varying eps during
        # training.
        self.accountant_type = "Amortized" # Moments or Amortized.

        ##############################################################################################################
        # Flags that control privacy spending during training.
        if preserve_privacy:
            self.eps = 1.0 # Start privacy spending for one epoch of training, used if accountant_type is Amortized.
#         self.end_eps = 1.0 # End privacy spending for one epoch of training, used if accountant_type is Amortized.
#         self.eps_saturate_epochs = 0 # Stop varying epsilon after eps_saturate_epochs. Set to 0 for constant eps of --eps. Used if accountant_type is Amortized.
            self.delta = 1e-5 # Privacy spending for training. Constant through training, used if accountant_type is Amortized.
            self.sigma = 4.0 # Noise sigma, used only if accountant_type is Moments
        else:
            self.eps = 0
            self.delta = 0
            self.sigma = 0
        ##############################################################################################################

        # Flags that control privacy spending for the pca projection
        # (only used if --projection_dimensions > 0).
#         self.pca_eps = 0.5 # Privacy spending for PCA, used if accountant_type is Amortized.
#         self.pca_delta = 0.005 # Privacy spending for PCA, used if accountant_type is Amortized.

#         self.pca_sigma = 7.0 # Noise sigma for PCA, used if accountant_type is Moments

        self.target_eps = "0.125,0.25,0.5,1,2,4,8" # Log the privacy loss for the target epsilon's. Only used when accountant_type is Moments.
        self.target_delta = 1e-5 # Maximum delta for --terminate_based_on_privacy.
        self.terminate_based_on_privacy = False # Stop training if privacy spent exceeds (max(--target_eps), --target_delta), even if --num_training_steps have not yet been completed.

        self.save_path = "/tmp/results" # Directory for saving model outputs.
        self.network_parameters = self.create_network_parameters()

        
    def create_network_parameters(self):
        network_parameters = utils.NetworkParameters()

        # If the ASCII proto isn't specified, then construct a config protobuf based
        # on 3 flags.
        network_parameters.input_size = self.image_size ** 2
        network_parameters.default_gradient_l2norm_bound = (
            self.default_gradient_l2norm_bound)
        if self.projection_dimensions > 0 and self.num_conv_layers > 0:
            raise ValueError("Currently you can't do PCA and have convolutions"
                             "at the same time. Pick one")
        
            # could add support for PCA after convolutions.
            # Currently BuildNetwork can build the network with conv followed by
            # projection, but the PCA training works on data, rather than data run
            # through a few layers. Will need to init the convs before running the
            # PCA, and need to change the PCA subroutine to take a network and perhaps
            # allow for batched inputs, to handle larger datasets.
        if self.num_conv_layers > 0:
            raise ValueError("Convolutional layers not supported in this demonstration. "
                            "See dp_mnist.py in differential_privacy folder for more options.")
        
        if self.projection_dimensions > 0:
            network_parameters.projection_type = "PCA"
            network_parameters.projection_dimensions = self.projection_dimensions
        for i in range(self.num_hidden_layers):
            hidden = utils.LayerParameters()
            hidden.name = "hidden%d" % i
            hidden.num_units = self.hidden_layer_num_units
            hidden.relu = True
            hidden.with_bias = False
            hidden.trainable = not self.freeze_bottom_layers
            network_parameters.layer_parameters.append(hidden)
        
        logits = utils.LayerParameters()
        logits.name = "logits"
        logits.num_units = 10
        logits.relu = False
        logits.with_bias = False
        network_parameters.layer_parameters.append(logits)
        return network_parameters
