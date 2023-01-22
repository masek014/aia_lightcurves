try:
    from . import aia_fmt_xml as afx
except ImportError:
    import aia_fmt_xml as afx
from dataclasses import dataclass

import astropy.units as u
import astropy.time
import copy
import functools
import logging
import math
import multiprocessing as mp
import multiprocessing.dummy as mpdummy
import os
import requests
import requests.exceptions as rex
import sys
import typing
import xmltodict

DATE_FMT = '%Y%m%d'
TIME_FMT = '%H%M%S'
DATETIME_FMT = f'{DATE_FMT}{TIME_FMT}'

@dataclass
class Config:
    read_timeout: u.s

# Configure this to be as you want.
cfg = Config(10 << u.s)
def debug_print(*args, **kwargs):
    logging.info(kwargs.get('sep', ' ').join(str(s) for s in args), **kwargs)

class DownloadResult(typing.NamedTuple):
    url: str
    file: str
    error: Exception=None
    @property
    def success(self):
        return (self.error is None)


@u.quantity_input
def download_aia_between(
    start: astropy.time.Time,
    end: astropy.time.Time,
    wavelengths: list[u.Angstrom],
    fits_out_dir: str,
    num_jobs: int=8,
    attempts: int=5
) -> list[DownloadResult]:
    '''
    download AIA fits files given the input args.
    returns: the list of filenames that were downloaded as DownloadResults
    '''
    all_urls = []
    successful = []
    failed = []
    # do this sequentially because it's fast
    debug_print('start find aia urls')
    for w in wavelengths:
        all_urls += build_aia_urls(start, end, w)
    debug_print(f'done find aia urls')

    download_wrapper = functools.partial(actual_download_files, fits_out_dir)

    # do this in parallel because it's slow
    tries = 0
    initial_num = len(all_urls)
    while tries < attempts:
        debug_print('start try downloads')
        with mpdummy.Pool(processes=num_jobs) as p:
            cur_downloaded = p.map(
                download_wrapper,
                all_urls,
                chunksize=math.ceil(len(all_urls) / num_jobs)
            )
        debug_print('done try downloads')

        failed = []
        for f in cur_downloaded:
            if f.success: successful.append(f)
            else: failed.append(f)

        all_successful = (failed == [])
        if all_successful: break
        debug_print('some failed ones:\n', failed)

        retry = [res.url for res in failed]
        failed = len(retry)
        errored_to_print = '\n\t' + '\n\t'.join(retry)
        print(f'{failed} / {initial_num} downloads failed.')
        print(f'retrying the following (attempt {1 + tries} / {attempts}): {errored_to_print}')
        all_urls = copy.deepcopy(retry)
        tries += 1

    return successful + failed


@u.quantity_input
def build_aia_urls(start: astropy.time.Time, end: astropy.time.Time, wavelength: u.Angstrom) -> list[str]:
    '''
    build up the download URLS given start, end times and a single wavelength
    '''
    query_str = build_query_string(start, end, wavelength)
    query_res = requests.request(
        'POST',
        afx.URL,
        headers=afx.XML_HEAD,
        data=query_str
    )

    file_ids = parse_file_ids(query_res.text)
    req_str = build_request_string(file_ids)
    urls_res = requests.request(
        'POST',
        afx.URL,
        headers=afx.XML_HEAD,
        data=req_str
    )
    return extract_urls(urls_res.text)


@u.quantity_input
def build_query_string(start: astropy.time.Time, end: astropy.time.Time, wavelength: u.Angstrom) -> str:
    '''
    given start, end times and a wavelength,
    return a properly-formatted AIA XML query string.
    '''
    start_str, end_str = start.strftime(DATETIME_FMT), end.strftime(DATETIME_FMT)
    wav_num = wavelength.to(u.Angstrom).value

    return afx.QUERY_FMT.format(
        start_time=start_str,
        end_time=end_str,
        wavelength=wav_num
    )


def build_request_string(file_ids: list[str]) -> str:
    '''
    build up AIA XML request string given a list of file IDs
    '''
    combined = '\n'.join(
        f'<fileid>{fid}</fileid>' for fid in file_ids
    )
    return afx.REQUEST_FMT.format(file_ids_parsed_xml=combined)


def parse_file_ids(query_response: str) -> list[str]:
    '''
    cut file IDs out of the AIA query response
    '''

    parsed = xmltodict.parse(query_response)
    wanted_items = (
        parsed
        ['soap:Envelope']
        ['soap:Body']
        ['VSO:QueryResponse']
        ['body']
        ['provideritem']
        [1:]
    )

    file_ids = []
    for item in wanted_items:
        records = item['record']['recorditem']
        for rec in records:
            file_ids.append(rec['fileid'])

    return file_ids


def parse_filename(content_disposition_header: str) -> str:
  '''
  get the filename from the response header from AIA
  this will probably break
  '''
  return content_disposition_header.split('"')[-2]


def extract_urls(request_result: str) -> list[str]:
    '''
    pull the actual URLs from the AIA data download request
    '''
    parsed = xmltodict.parse(request_result)
    urlz = (
        parsed
        ['soap:Envelope']
        ['soap:Body']
        ['VSO:GetDataResponse']
        ['body']
        ['getdataresponseitem']
        ['getdataitem']
        ['dataitem']
    )
    return list(uu['url'] for uu in urlz)


def actual_download_files(output_directory: str, url: str) -> DownloadResult:
    '''
    download AIA files
    return: (file name output, success or not)
    '''
    debug_print(f'wait time is {cfg.read_timeout}')
    debug_print('gotcha:', url)
    full_fn = '🥲'
    try:
        with requests.get(url, stream=True, timeout=cfg.read_timeout.to(u.s).value) as res:
            res.raise_for_status()
            fn = parse_filename(res.headers['Content-Disposition'])
            file_size = int(res.headers['Content-Length'])

            full_fn = f'{output_directory}/{fn}'
            if os.path.exists(full_fn) and os.stat(full_fn).st_size == file_size:
                debug_print('file already exists and is downloaded:', full_fn)
                debug_print()
                return DownloadResult(url, full_fn)

            with open(full_fn, 'wb') as f:
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    except (rex.RequestException, rex.HTTPError) as e:
        return DownloadResult(url, full_fn, e)

    return DownloadResult(url, full_fn)


def test():
    print('request test')
    start = astropy.time.Time('2019-04-03T17:40:00.0')
    end = astropy.time.Time('2019-04-03T17:50:00.0')
    wavelengths = [171 * u.Angstrom]

    out_dir = 'test-manual-download'
    os.makedirs(out_dir, exist_ok=True)
    ret = download_aia_between(
        start=start,
        end=end,
        wavelengths=wavelengths,
        fits_out_dir=out_dir,
        num_jobs=8,
        attempts=3
    )
    if all(r.success for r in ret):
        print('all good')
    else:
        print('some failed')
    return ret


def timed_test():
    from datetime import datetime

    start_time = datetime.now()
    ret = test()
    end_time = datetime.now()

    print(f'Duration: {end_time - start_time}')
    return ret


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    timed_test()
