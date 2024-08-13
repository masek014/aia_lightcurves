XML_HEAD = {
    'Content-Type': 'text/xml; charset=utf-8'
}

URL = 'http://vso.nso.edu/cgi-bin/VSO/PROD/vsoi_wsdl.cgi'

QUERY_FMT = '''<?xml version='1.0' encoding='utf-8'?>
<soap-env:Envelope xmlns:soap-env="http://schemas.xmlsoap.org/soap/envelope/" xmlns:VSO="http://virtualsolar.org/VSO/VSOi">
  <soap-env:Body>
    <VSO:Query>
      <body>
        <block>
          <time>
            <start>{start_time}</start>
            <end>{end_time}</end>
          </time>
          <instrument>AIA</instrument>
          <wave>
            <wavemin>{wavelength:.1f}</wavemin>
            <wavemax>{wavelength:.1f}</wavemax>
            <waveunit>Angstrom</waveunit>
          </wave>
        </block>
      </body>
    </VSO:Query>
  </soap-env:Body>
</soap-env:Envelope>'''

REQUEST_FMT = '''<?xml version='1.0' encoding='utf-8'?>
<soap-env:Envelope xmlns:soap-env="http://schemas.xmlsoap.org/soap/envelope/" xmlns:VSO="http://virtualsolar.org/VSO/VSOi">
  <soap-env:Body>
    <VSO:GetData>
      <body>
        <request>
          <method>
            <methodtype>URL-FILE_Rice</methodtype>
            <methodtype>URL-FILE</methodtype>
            <methodtype>URL-packaged</methodtype>
            <methodtype>URL-TAR_GZ</methodtype>
            <methodtype>URL-ZIP</methodtype>
            <methodtype>URL-TAR</methodtype>
            <methodtype>URL</methodtype>
          </method>
          <info>
            <email>sunpied</email>
          </info>
          <datacontainer>
            <datarequestitem>
              <provider>JSOC</provider>
              <fileiditem>
{file_ids_parsed_xml}
              </fileiditem>
            </datarequestitem>
          </datacontainer>
        </request>
      </body>
    </VSO:GetData>
  </soap-env:Body>
</soap-env:Envelope>
'''
