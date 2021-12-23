import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

import numpy as np


from util import _clean_attrs, _sanitize_data_for_writing


class BulkFast5(h5py.File):
    """Class for reading data from a bulk fast5 file"""

    __tracking_path__ = '/UniqueGlobalKey/tracking_id'
    __pore_model_old__ = 'Meta/User/pore_model'
    __pore_model_new__ = 'Meta/User/analysis_conf'
    __context_path__ = '/UniqueGlobalKey/context_tags/'
    __intermediate_data__ = '/IntermediateData/'
    __voltage_meta__ = '/Device/VoltageMeta'
    __voltage_data__ = '/Device/MetaData'
    __channel_meta__ = '/IntermediateData/Channel_{}/Meta'
    __multiplex_data__ = '/MultiplexData/Channel_{}/Multiplex'

    __raw_data__ = "Raw/Channel_{}/Signal"
    __raw_meta__ = "Raw/Channel_{}/Meta"
    __event_data__ = "/IntermediateData/Channel_{}/Events"
    __read_data__ = "/IntermediateData/Channel_{}/Reads"
    __state_data__ = "/StateData/Channel_{}/States"

    # The below refers to MinION Mk1 ASIC, may change in future
    __mk1_asic_mux_states__ = {
        'common_voltage_1': 1,
        'common_voltage_2': 2,
        'common_voltage_3': 3,
        'common_voltage_4': 4,
        'gnd': 15,
        'gnd_through_resistor': 14,
        'open_pore': 0,
        'test_current_1': 10,
        'test_current_2': 11,
        'test_current_3': 12,
        'test_current_4': 13,
        'test_current_open_pore': 5,
        'unblock_voltage_1': 6,
        'unblock_voltage_2': 7,
        'unblock_voltage_3': 8,
        'unblock_voltage_4': 9
    }

    def __init__(self, filename, mode='r'):
        """Create an BulkFast5 instance.

        :param filename: path to a bulk fast5 file.
        :param mode: h5py opening mode.
        """

        super(BulkFast5, self).__init__(filename, mode)
        if mode == 'r':
            data = self[self.__intermediate_data__]
            self.channels = sorted([int(name.strip('Channel_')) for name in data.keys()])
            self.parsed_exp_history = None # we parse the history lazily

            # Parse experimental metadata
            self.exp_metadata = dict()
            for path in (self.__tracking_path__, self.__context_path__):
                try:
                    self.exp_metadata.update(_clean_attrs(self[path].attrs))
                except KeyError:
                    raise KeyError('Cannot read summary from {}'.format(path))

            # This should be safe
            try:
                self.sample_rate = float(self['Meta'].attrs['sample_rate'])
            except:
                self.sample_rate = float(self.get_metadata(self.channels[0])['sample_rate'])


    def get_metadata(self, channel):
        """Get the metadata for the specified channel.

        Look for first for events metadata, and fall-back on raw metadata, returning an empty dict if neither could be found."""
        if hasattr(self, '_cached_metadata'):
            if channel in self._cached_metadata:
                return self._cached_metadata[channel]
        else:
            self._cached_metadata = {}

        if self.__channel_meta__.format(channel) in self:
            meta = _clean_attrs(self[self.__channel_meta__.format(channel)].attrs)
        elif self.has_raw(channel): # use raw meta data
            meta = _clean_attrs(self[self.__raw_meta__.format(channel)].attrs)
        else:
            meta = {}

        self._cached_metadata[channel] = meta
        return meta


    def get_tracking_meta(self):
        """Get tracking meta data"""
        return _clean_attrs(self[self.__tracking_path__].attrs)


    def get_context_meta(self):
        """Get context meta"""
        return _clean_attrs(self[self.__context_path__].attrs)


    def has_raw(self, channel):
        """Return True if there is raw data for this channel."""
        raw_location = self.__raw_data__.format(channel)
        return self._has_data(raw_location)


    def has_reads(self, channel):
        """Return True if there is read data for this channel."""
        read_location = self.__read_data__.format(channel)
        return self._has_data(read_location)


    def has_states(self, channel):
        """Return True if there is State data for this channel."""
        state_location = self.__state_data__.format(channel)
        return self._has_data(state_location)


    def _has_data(self, location):
        """Return true if the given data path exists

        :param location: str, path with fast5.
        """
        if hasattr(self, '_cached_paths'):
            if location in self._cached_paths:
                return self._cached_paths[location]
        else:
            self._cached_paths = {}

        location_split = location.split('/')
        folder = '/'.join(location_split[:-1])
        name = location_split[-1]
        present = folder in self and name in self[folder].keys()
        self._cached_paths[location] = present
        return present


    def _time_interval_to_index(self, channel, times):
        """Translate a tuple of (start_sec, end_sec) to an index."""
        start_sec, end_sec = times
        start = self._seconds_to_index(channel, start_sec)
        end = self._seconds_to_index(channel, end_sec)
        return (start, end)


    def _seconds_to_index(self, channel, time):
        """Translate a point in time to an index."""
        if time is None:
            return None

        return int(time * float(self.sample_rate))


    def _scale(self, channel, data):
        """Scale event data if necessary, else return unchanged.

        If event metadata can't be found, assume events don't need scaling."""

        meta_data = self.get_metadata(channel)

        if 'scaling_used' not in meta_data or meta_data.get('scaling_used'):
            return data
        else:
            channel_scale = meta_data['range'] / meta_data['digitisation']
            channel_offset = meta_data['offset']
            data['mean'] = (data['mean'] + channel_offset) * channel_scale
            return data


    def get_raw(self, channel, times=None, raw_indices=(None, None), use_scaling=True):
        """If available, parse channel raw data.

        :param channel: channel number int
        :param times: tuple of floats (start_second, end_second)
        :param raw_indices: tuple of ints (start_index, end_index)
        :param use_scaling: if True, scale the current level

        .. note::
            Exactly one of the slice keyword arguments needs to be specified,
            as the method will override them in the order of times
            > raw_indices.
        """

        if not self.has_raw(channel):
            raise KeyError('Channel {} does not contain raw data.'.format(channel))

        if times is not None:
            raw_indices = self._time_interval_to_index(channel, times)

        raw_data = self.__raw_data__.format(channel)
        data = self[raw_data][raw_indices[0]:raw_indices[1]]

        if use_scaling:
            meta_data = self.get_metadata(channel)
            raw_unit = meta_data['range'] / meta_data['digitisation']
            data = (data + meta_data['offset']) * raw_unit

        return data


    def _add_attrs(self, data, location, convert=None):
        """Convenience method for adding attrs to a possibly new group.
        :param data: dict of attrs to add
        :param location: hdf path
        :param convert: function to apply to all dictionary values
        """
        self.__add_attrs(self, data, location, convert=None)


    @staticmethod
    def __add_attrs(self, data, location, convert=None):
        """Implementation of _add_attrs as staticmethod. This allows
        functionality to be used in .New() constructor but is otherwise nasty!
        """
        if location not in self:
            self.create_group(location)
        attrs = self[location].attrs
        for k, v in data.items():
            if convert is not None:
                attrs[_sanitize_data_for_writing(k)] = _sanitize_data_for_writing(convert(v))
            else:
                attrs[_sanitize_data_for_writing(k)] = _sanitize_data_for_writing(v)


    def _add_numpy_table(self, data, location):
        data = _sanitize_data_for_writing(data)
        self.create_dataset(location, data=data, compression=True)


    @classmethod
    def New(cls, fname, read='a', tracking_id={}, context_tags={}, channel_id={}):
        """Construct a fresh bulk file, with meta data written to
        standard locations. There is currently no checking this meta data.
        TODO: Add meta data checking.

        """

        # Start a new file, populate it with meta
        with h5py.File(fname, 'w') as h:
            h.attrs[_sanitize_data_for_writing('file_version')] = _sanitize_data_for_writing(1.0)
            for data, location in zip(
                [tracking_id, context_tags],
                [cls.__tracking_path__, cls.__context_path__]
            ):
                # see cjw's comment in fast5.py:
                # 'no idea why these must be str, just following ossetra'
                cls.__add_attrs(h, data, location, convert=str)

        # return instance from new file
        return cls(fname, read)


    def set_raw(self, raw, channel, meta=None):
        """Set the raw data in file.

        :param raw: raw data to add
        :param channel: channel number
        """
        req_keys = ['description', 'digitisation', 'offset', 'range',
                    'sample_rate']

        meta = {k:v for k,v in meta.items() if k in req_keys}
        if len(meta.keys()) != len(req_keys):
            raise KeyError(
                'Raw meta data must contain keys: {}.'.format(req_keys)
            )

        raw_folder = '/'.join(self.__raw_data__.format(channel).split('/')[:-1])
        raw_data_path = self.__raw_data__.format(channel)
        self._add_attrs(meta, raw_folder)
        self[raw_data_path] = raw


    def set_voltage(self, data, meta):
        req_keys = ['description', 'digitisation', 'offset', 'range',
                    'sample_rate']
        meta = {k:v for k,v in meta.items() if k in req_keys}
        if len(meta.keys()) != len(req_keys):
            raise KeyError(
                'Raw meta data must contain keys: {}.'.format(req_keys)
            )

        self._add_attrs(meta, self.__voltage_meta__)
        dtype = np.dtype([('bias_voltage', np.int16)])
        self._add_numpy_table(
            data.astype(dtype, copy=False), self.__voltage_data__

        )
