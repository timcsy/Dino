from eventstreaming import stream

stream.start()

stream.record_io_event('record.json')

stream.close()