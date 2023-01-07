from . import aia_fmt_xml as afx

import astropy.units as u
import astropy.time
import functools
import logging
import math
import multiprocessing as mp
import os
import requests
import requests.exceptions
import sys
import typing
import xmltodict

# configure this to be as you want
read_timeout = 10 << u.s

@u.quantity_input
def download_aia_between(
    start: astropy.time.Time,
    end: astropy.time.Time,
    wavelengths: list[u.Angstrom],
    fits_out_dir: str,
    num_jobs: int=8,
    attempts: int=5
) -> list[str]:
    '''
    download AIA fits files given the input args.
    returns: the list of filenames that were downloaded
    '''
    files_downloaded = []
    all_urls = []
    # do this sequentially because it's fast
    logging.debug('start find aia urls')
    for w in wavelengths:
        all_urls += build_aia_urls(start, end, w)
    logging.debug(f'done find aia urls:\n{all_urls}')

    download_wrapper = functools.partial(actual_download_files, fits_out_dir)

    # do this in parallel because it's slow
    tries = 0
    while tries < attempts:
        logging.debug('start try downloads')
        with mp.Pool(processes=num_jobs) as p:
            files_downloaded = p.map(
                download_wrapper,
                all_urls,
                chunksize=math.ceil(len(all_urls) / num_jobs)
            )
        logging.debug('done try downloads')

        errored = [all_urls[i] for (i, res) in enumerate(files_downloaded) if not res[1]]
        if not errored: break

        failed = len(errored)
        initial = len(all_urls)
        errored_to_print = '\n' + '\t\n'.join(errored)
        logging.info(f'{failed} / {initial} downloads failed.')
        logging.info(f'retrying the following (attempt {1 + tries} / {attempts}): {errored_to_print}')
        all_urls = errored
        tries += 1

    return files_downloaded


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
    TIME_FMT = '%Y%m%d%H%M%S'
    start_str, end_str = start.strftime(TIME_FMT), end.strftime(TIME_FMT)
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


class DownloadResult(typing.NamedTuple):
    file: str
    success: bool

def actual_download_files(output_directory: str, url: str) -> list[DownloadResult]:
    '''
    download AIA files
    return: (file name output, success or not)
    '''
    logging.debug('gotcha:', url)
    full_fn = '🥲'
    try:
        with requests.get(url, stream=True, timeout=read_timeout.to(u.s).value) as res:
            res.raise_for_status()
            fn = parse_filename(res.headers['Content-Disposition'])
            full_fn = f'{output_directory}/{fn}'
            with open(full_fn, 'wb') as f:
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    except requests.exceptions.ReadTimeout:
        return DownloadResult(full_fn, False)

    return DownloadResult(full_fn, True)


def test():
    print('request test')
    start = astropy.time.Time('2020-02-01T03:00:00')
    end = astropy.time.Time('2020-02-01T03:01:00')
    wavelengths = [171 * u.Angstrom]

    out_dir = 'test-manual-download'
    os.makedirs(out_dir, exist_ok=True)
    ret = download_aia_between(
        start=start,
        end=end,
        wavelengths=wavelengths,
        fits_out_dir=out_dir,
        num_jobs=8,
        attempts=5
    )
    return ret


def timed_test():
    from datetime import datetime
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    start_time = datetime.now()
    ret = test()
    end_time = datetime.now()

    print(f'Duration: {end_time - start_time}')
    return ret


if __name__ == '__main__':
    timed_test()
