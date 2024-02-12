import numpy as np
import scipy
from deepsort import iou_matching
from deepsort.iou_matching import linear_assignment
from deepsort.tracker import Track

from algorithm.DetectedObject import DetectedObject
from algorithm.flow_conditions import rot_mat_from_river_velocity
from algorithm.matching.linear_assignment import matching_cascade, min_cost_matching


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, conf: dict):
        """Initialize Kalman filter.

        Args:
            obj_velocity_initalization (tuple[float, float], optional): Initialization of velocity of objects
                in x and y direction. Defaults to (0., 0.).
        """
        self.conf = conf
        # adapt the initial velocity to the river velocity
        self.obj_velocity_initalization = [
            v / 4.0 for v in self.conf["river_pixel_velocity"]
        ]

        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.array([*self.obj_velocity_initalization, 0, 0])
        mean = np.r_[mean_pos, mean_vel]

        std_trace = (
            np.array(self.conf["kalman_std_obj_initialization_trace"])
            * self.conf["kalman_std_obj_initialization_factor"]
        )
        bbox_height_scaling_selection = np.array([1, 1, 0, 1, 1, 1, 0, 1])
        std_trace = std_trace * (
            bbox_height_scaling_selection * measurement[3]
            + 1
            - bbox_height_scaling_selection
        )
        covariance = np.diag(np.square(std_trace))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_trace = (
            np.array(self.conf["kalman_std_process_noise_trace"])
            * self.conf["kalman_std_obj_initialization_factor"]
        )
        bbox_height_scaling_selection = np.array([1, 1, 0, 1, 1, 1, 0, 1])
        std_trace = std_trace * (
            bbox_height_scaling_selection * mean[3] + 1 - bbox_height_scaling_selection
        )
        motion_cov = np.diag(np.square(std_trace))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std_trace = (
            np.array(self.conf["kalman_std_mmt_noise_trace"])
            * self.conf["kalman_std_obj_initialization_factor"]
        )
        bbox_height_scaling_selection = np.array([1, 1, 0, 1])
        std_trace = std_trace * (
            bbox_height_scaling_selection * mean[3] + 1 - bbox_height_scaling_selection
        )
        # rotate the measurement covariance matrix into the river flow direction
        if self.conf["kalman_rotate_mmt_noise_in_river_direction"]:
            xy_std = np.dot(self.rot_mat.T, np.diag(std_trace[:2]) ** 2)
            ah_std = np.diag(std_trace[2:]) ** 2
            innovation_cov = scipy.linalg.block_diag(xy_std, ah_std)
        else:
            innovation_cov = np.diag(std_trace) ** 2

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # limit velocity to maximum 2 * v_river to prevent jumps of tracked objects
        v = new_mean[4:6]
        v_norm = np.linalg.norm(v)
        if v_norm > 2 * np.linalg.norm(self.conf["river_pixel_velocity"]):
            v = v / v_norm * 2 * np.linalg.norm(self.conf["river_pixel_velocity"])
            new_mean[4:6] = v

        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    @property
    def rot_mat(self):
        return rot_mat_from_river_velocity(self.conf)


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    obj_velocity_initalization : tuple[float, float]
        Initialization of velocity of objects in x and y direction.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(
        self,
        metric,
        conf: dict,
    ) -> None:
        self.metric = metric
        self.conf = conf
        self.max_iou_distance = self.conf["kalman_max_iou_distance"]
        self.max_age = self.conf["kalman_max_age"]
        self.n_init = self.conf["kalman_n_init"]

        self.kf = KalmanFilter(conf)
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: dict[int:DetectedObject]):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        ds_detections = [d.deepsort_detection for d in detections.values()]
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, ds_detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(ds_detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(features, targets, active_targets)

    def _match(self, detections: dict[int, DetectedObject]):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = [dets[i].feature for i in detection_indices]
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices
            )

            return cost_matrix

        ds_detections = [d.deepsort_detection for d in detections.values()]

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            ds_detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            ds_detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                detection.feature,
            )
        )
        self._next_id += 1


def filter_detections(
    detections: dict[int:DetectedObject],
    tracker: Tracker,
):
    tracker.predict()
    tracker.update(detections)


def tracks_to_object_history(
    tracks: list[Track],
    object_history: dict[int, DetectedObject],
    frame_number: int,
) -> dict[int, DetectedObject]:
    for track in tracks:
        if track.is_confirmed():
            obj = DetectedObject(track.track_id, track.to_tlwh(), frame_number)
            if track.track_id not in object_history.keys():
                object_history[track.track_id] = obj
            else:
                object_history[track.track_id].update_object(obj)
    return object_history
