# Requirements Document

## Introduction

Bir fonun "Fon İçeriği" panelinde ve fon ile ilgili diğer alt fon listelerinde, fon kategorisindeki pozisyonlar şu anda statik metin olarak görüntüleniyor. Kullanıcı, bir ana fonun içinde tuttuğu alt fonların (örneğin IIE içindeki IIP/ITH/IMR/PRE; bir fon-sepetinin alt fonları; ya da kullanıcının daha önce hiç açmadığı fonlar) getirisini ve detaylarını da kolayca görmek istiyor.

Bu özellik; alt fon pozisyon satırlarını tıklanabilir hâle getirecek, mevcut Fund Detail tam sayfasına simetrik bir geçiş sağlayacak ve TEFAS'ta işlem gören (TEFAS-tradable) fonlar ile diğerlerini görsel olarak ayıracaktır. Tıklanabilirlik kararı backend tarafından üretilen `tefas_tradable` bayrağına göre verilir; frontend bu bayrağı okur ve buna göre satırı aktif veya devre dışı render eder. Hover üzerine fund detail ve fund performance verisi prefetch edilir, böylece tıklamada detay sayfası anında dolar.

## Glossary

- **Frontend**: `frontend/src/` altındaki React + TypeScript uygulaması.
- **Backend**: `app/` altındaki FastAPI tabanlı RAG-FIN servisi.
- **Fund_Service**: `app/fund_service.py` modülü ve içindeki holdings normalize / response oluşturma akışı.
- **Fund_Holdings_Panel**: Fon detay sayfasındaki "Fon İçeriği" panelini render eden frontend bileşeni (örn. `FundHoldingsPanel`).
- **Fund_Position_Row**: `Fund_Holdings_Panel` içindeki tek bir pozisyonu (asset_type, asset_code, weight) temsil eden satır.
- **Fund_Position_Category**: Bir pozisyonun türü; "fund", "stock", "bond", "cash" gibi değerler. Bu özellik yalnızca `Fund_Position_Category` değeri "fund" olan satırları etkiler.
- **Fund_Detail_Page**: Fon detay tam sayfası (frontend `FundDetail` / `FundsPage` üzerinden açılan görünüm).
- **Fund_Snapshot_Index**: Backend `/funds` listesinin döndürdüğü TEFAS açık fonlar koleksiyonu (varsayılan olarak `TEFAS_OPEN_ONLY=1` ile süzülmüş).
- **TEFAS_Tradable_Flag**: Bir alt fon pozisyonu için backend tarafından üretilen ve değer olarak `true`, `false` ya da yok (alan tanımsız) alabilen boolean bayrak. JSON alan adı: `tefas_tradable`.
- **Restricted_Data_Notice**: Fund_Detail_Page açıldığında, eksik alan tespit edildiğinde gösterilen kısa uyarı şeridi.
- **Hover_Prefetch_Debounce_Ms**: Bir Fund_Position_Row üzerinde hover sırasında prefetch tetiklenmeden önce beklenen sabit gecikme süresi; değeri 150 milisaniyedir.
- **Prefetch_Service**: Frontend tarafında fund detail ve fund performance sorgularını arka planda doldurmak için kullanılan react-query prefetch mekanizması.
- **Fund_Detail_Query_Key**: react-query cache anahtarı; `["fund-detail", asset_code]` formundadır. Aynı `asset_code` için tek bir cache girdisini temsil eder.
- **Fund_Performance_Query_Key**: react-query cache anahtarı; `["fund-performance", asset_code, period]` formundadır. Farklı `period` değerleri (örn. "1Y", "3M") farklı cache girdileri olarak değerlendirilir.
- **Default_Performance_Prefetch_Period**: Hover prefetch sırasında `Fund_Performance_Query_Key` için kullanılan varsayılan dönem değeri; sabit değeri `"1Y"` (son 1 yıl) dizgesidir.
- **Max_Concurrent_Hover_Prefetch**: Aynı anda in-flight durumda olabilecek hover-prefetch network isteklerinin üst sınırı; değeri 2'dir.
- **Holdings_Response_Schema_Version**: `app/fund_service.py` içindeki `_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION` sabiti; response cache anahtarına dahil edilen sürüm numarası.
- **KAP_Holdings_Parse_Version**: KAP holdings parser çıktısının cache versiyon numarası; parser davranışı değişmediğinde sabit kalır.

## Requirements

### Requirement 1: Fund kategorisindeki alt fon pozisyonlarını tıklanabilir yapma

**User Story:** Bir kullanıcı olarak, bir fonun Fon İçeriği panelinde gördüğüm alt fon pozisyonlarına tıklayarak o alt fonun detay sayfasına gidebilmek istiyorum, böylece daha önce hiç açmadığım bir alt fonun getirisini de aynı akışta inceleyebilirim.

#### Acceptance Criteria

1. WHEN bir Fund_Position_Row render edilir AND `Fund_Position_Category` değeri "fund" AND `TEFAS_Tradable_Flag` değeri `true`, THE Fund_Holdings_Panel SHALL satırı tıklanabilir bir kontrol olarak (görsel olarak link/buton stiliyle) render eder.
2. WHEN tıklanabilir bir Fund_Position_Row üzerinde fare ile sol tıklama gerçekleşir, THE Fund_Holdings_Panel SHALL `onOpenFund(asset_code, "overview")` callback'ini tam olarak bir kez çağırır.
3. WHEN tıklanabilir bir Fund_Position_Row üzerinde klavye ile `Enter` veya `Space` tuşuna basılır AND satır odaklanmış durumda, THE Fund_Holdings_Panel SHALL fare tıklamasıyla aynı `onOpenFund(asset_code, "overview")` callback'ini çağırır.
4. WHEN `onOpenFund` callback'i Fund_Holdings_Panel tarafından çağrılır, THE Frontend SHALL Fund_Detail_Page'i parametre olarak verilen `asset_code` için yükler ve URL state'ini fon detay görünümüne günceller.
5. THE Fund_Holdings_Panel SHALL `Fund_Position_Category` değeri "fund" dışındaki satırlar için bu özelliğin davranışını değiştirmez ve mevcut hisse pozisyonu davranışını (örn. `onOpenTicker`) bozmaz.

### Requirement 2: TEFAS-tradable bayrağının backend tarafında üretilmesi

**User Story:** Bir frontend geliştiricisi olarak, bir alt fon pozisyonunun tıklanabilir olup olmayacağına karar vermek için backend'den net bir bayrak almak istiyorum, böylece UI'de ek bir liste sorgusu yapmadan satırı doğru render edebilirim.

#### Acceptance Criteria

1. WHEN Fund_Service bir holdings yanıtı için pozisyonları normalize eder AND pozisyonun `Fund_Position_Category` değeri "fund", THE Fund_Service SHALL pozisyon nesnesine `TEFAS_Tradable_Flag` (`tefas_tradable`) alanını ekler.
2. WHEN normalize edilen pozisyonun `asset_code` değeri Fund_Snapshot_Index içinde mevcut, THE Fund_Service SHALL `TEFAS_Tradable_Flag` değerini `true` olarak ayarlar.
3. WHEN normalize edilen pozisyonun `asset_code` değeri Fund_Snapshot_Index içinde mevcut değil AND `Fund_Position_Category` değeri "fund", THE Fund_Service SHALL `TEFAS_Tradable_Flag` değerini `false` olarak ayarlar.
4. WHERE `Fund_Position_Category` değeri "fund" değilse, THE Fund_Service SHALL `TEFAS_Tradable_Flag` alanını yanıta eklemez.
5. THE Fund_Service SHALL `TEFAS_Tradable_Flag` türetimi için Fund_Snapshot_Index'i her holdings yanıtı oluşturulduğunda tek seferlik bir referans hâlinde kullanır ve pozisyon başına tek tek harici sorgu yapmaz.
6. WHEN Fund_Snapshot_Index Fund_Service tarafında erişilemez (örn. boş veya hata) AND holdings yanıtı yine de oluşturulmak zorunda, THE Fund_Service SHALL "fund" kategorisindeki pozisyonların `TEFAS_Tradable_Flag` değerini `false` olarak ayarlar ve uyarıyı backend log'a yazar.

### Requirement 3: TEFAS'ta işlem görmeyen alt fonların disabled olarak gösterilmesi

**User Story:** Bir kullanıcı olarak, TEFAS'ta açık olarak işlem görmeyen (kapalı GSYF, niş veya erişilemez) alt fonların satırını görmek ama yanlışlıkla bunların detayını açmaya çalışmamak istiyorum, böylece neden tıklayamadığımı anlayabilirim.

#### Acceptance Criteria

1. WHEN bir Fund_Position_Row render edilir AND `Fund_Position_Category` değeri "fund" AND `TEFAS_Tradable_Flag` değeri `false`, THE Fund_Holdings_Panel SHALL satırı listede tutar.
2. WHILE bir Fund_Position_Row `TEFAS_Tradable_Flag` değeri `false` ile render edilmiş durumda, THE Fund_Holdings_Panel SHALL satırı görsel olarak devre dışı stil ile (azaltılmış kontrast, imleç `not-allowed`) gösterir.
3. WHILE bir Fund_Position_Row devre dışı durumda, THE Fund_Holdings_Panel SHALL satır üzerindeki tıklama, `Enter` ve `Space` etkileşimleri için `onOpenFund` callback'ini çağırmaz.
4. WHEN kullanıcı devre dışı bir Fund_Position_Row üzerine fare ile en az 400 milisaniye süreyle hover eder VEYA satıra klavye ile odaklanır, THE Fund_Holdings_Panel SHALL "Bu fon TEFAS'ta açık olarak işlem görmediği için detay sayfası açılamıyor" mesajını içeren bir tooltip gösterir.
5. THE Fund_Holdings_Panel SHALL devre dışı Fund_Position_Row'lar için DOM düzeyinde `aria-disabled="true"` özniteliğini ayarlar.

### Requirement 4: Davranışın diğer alt fon listelerinde de geçerli olması

**User Story:** Bir kullanıcı olarak, alt fon listelerini fonun her gösterildiği yerde aynı kuralla görmek istiyorum, böylece bir yerde tıklayabildiğim fonu başka bir yerde de aynı şekilde tıklayabileyim ya da aynı sebeple devre dışı görebileyim.

#### Acceptance Criteria

1. WHERE Frontend içinde bir alt fon listesi `Fund_Position_Row` veri modeline dayalı olarak render edilir, THE ilgili bileşen SHALL Requirement 1, Requirement 3 ve Requirement 7 kurallarını aynı şekilde uygular.
2. WHERE bir alt fon listesi `onOpenFund` callback'i prop'unu almıyor, THE ilgili bileşen SHALL "fund" kategorisindeki satırları görsel olarak devre dışı stille gösterir ve tıklama / klavye etkinliklerini yutar.
3. THE Frontend SHALL alt fon satırı render mantığını paylaşılan tek bir bileşen veya hook üzerinden yeniden kullanır ve tıklama davranışı kuralını her panelde tekrar implemente etmez.

### Requirement 5: Hover üzerine prefetch ile anında detay açılışı

**User Story:** Bir kullanıcı olarak, bir alt fon satırının üzerine geldiğimde detay sayfasının arka planda hazırlanmasını istiyorum, böylece tıkladığımda fon detayı yükleme bekletmeden açılsın.

#### Acceptance Criteria

1. WHEN tıklanabilir bir Fund_Position_Row üzerinde fare hover'ı `Hover_Prefetch_Debounce_Ms` değerinden uzun süreyle kesintisiz devam eder, THE Prefetch_Service SHALL ilgili `asset_code` için `Fund_Detail_Query_Key` (`["fund-detail", asset_code]`) altında fund detail sorgusunu arka planda doldurur.
2. WHEN tıklanabilir bir Fund_Position_Row üzerinde klavye odağı `Hover_Prefetch_Debounce_Ms` değerinden uzun süreyle kesintisiz devam eder, THE Prefetch_Service SHALL ilgili `asset_code` için `Fund_Detail_Query_Key` (`["fund-detail", asset_code]`) altında fund detail sorgusunu arka planda doldurur.
3. WHEN tıklanabilir bir Fund_Position_Row üzerinde fare hover'ı veya klavye odağı `Hover_Prefetch_Debounce_Ms` değerinden uzun süreyle kesintisiz devam eder, THE Prefetch_Service SHALL ilgili `asset_code` için `Fund_Performance_Query_Key` (`["fund-performance", asset_code, Default_Performance_Prefetch_Period]`) altında fund performance sorgusunu `Default_Performance_Prefetch_Period` değeri (`"1Y"`) ile arka planda doldurur.
4. IF kullanıcı `Hover_Prefetch_Debounce_Ms` süresi dolmadan satırdan ayrılır (`mouseleave`) VEYA odak başka bir öğeye geçer (`blur`), THEN THE Prefetch_Service SHALL bu satır için kayıtlı debounce zamanlayıcısını iptal eder AND ilgili `Fund_Detail_Query_Key` ve `Fund_Performance_Query_Key` için hiçbir network isteği başlatmaz.
5. WHEN aynı oturum içinde belirli bir react-query cache anahtarı için (örn. aynı `Fund_Detail_Query_Key` veya aynı `(asset_code, period)` çiftine sahip `Fund_Performance_Query_Key`) prefetch son 60 saniye içinde tamamlanmış, THE Prefetch_Service SHALL aynı cache anahtarı için yeni bir prefetch isteği tetiklemez.
6. WHERE iki farklı `Fund_Performance_Query_Key` girdisi yalnızca `period` değerinde farklılaşır (örn. `["fund-performance", "AEK", "1Y"]` ve `["fund-performance", "AEK", "3M"]`), THE Prefetch_Service SHALL bu girdileri ayrı cache anahtarları olarak değerlendirir AND birinin son 60 saniye içinde tamamlanmış olması diğerinin prefetch'ini engellemez.
7. WHILE Prefetch_Service tarafında halihazırda `Max_Concurrent_Hover_Prefetch` (2) hover-prefetch network isteği in-flight durumda AND yeni bir hover-prefetch tetiklenmek üzere, THE Prefetch_Service SHALL henüz network çağrısı başlatılmamış (kuyrukta bekleyen) en eski hover-prefetch görevini iptal eder AND yeni hover-prefetch görevini onun yerine kuyruğa alır.
8. WHILE bir hover-prefetch network isteği zaten başlatılmış (in-flight) durumda, THE Prefetch_Service SHALL `Max_Concurrent_Hover_Prefetch` sınırı aşılsa dahi başlatılmış bu isteği iptal etmez ve doğal olarak tamamlanmasına izin verir.
9. WHILE bir Fund_Position_Row devre dışı durumda (`TEFAS_Tradable_Flag` değeri `false` VEYA Requirement 6'ya göre `tefas_tradable` alanı yanıtta yok), THE Prefetch_Service SHALL bu satır için herhangi bir debounce zamanlayıcısı oluşturmaz AND `Fund_Detail_Query_Key` veya `Fund_Performance_Query_Key` altında herhangi bir prefetch network isteği başlatmaz.

### Requirement 6: Eski yanıtlarla geriye dönük uyumluluk

**User Story:** Bir kullanıcı olarak, backend güncellenmeden önce oluşmuş cache yanıtları döndürdüğünde de UI'in tutarlı şekilde çalışmasını istiyorum, böylece yarım yamalak bir arayüzle karşılaşmayayım.

#### Acceptance Criteria

1. WHEN bir Fund_Position_Row "fund" kategorisinde AND `TEFAS_Tradable_Flag` alanı yanıtta yok, THE Fund_Holdings_Panel SHALL satırı Requirement 3'teki devre dışı görünümle render eder ve `onOpenFund` callback'ini çağırmaz.
2. WHEN Fund_Service Holdings_Response_Schema_Version değerini `TEFAS_Tradable_Flag` alanını üretmek üzere artırır, THE Backend SHALL mevcut response cache girdilerini bu sürüm artışı sayesinde geçersiz kılar.
3. THE Fund_Service SHALL `TEFAS_Tradable_Flag` desteğini eklerken KAP_Holdings_Parse_Version değerini değiştirmez.
4. WHEN frontend bir holdings yanıtında "fund" kategorisinde bir satır görür AND yanıttaki herhangi bir "fund" satırının `TEFAS_Tradable_Flag` değeri tanımlı, THE Frontend SHALL sayfa içindeki diğer "fund" satırlarının `TEFAS_Tradable_Flag` değerini de yanıttaki değerleriyle yorumlar ve tek bir varsayılan değere sabitlemez.

### Requirement 7: Klavye ve okuyucular için erişilebilirlik

**User Story:** Klavye veya ekran okuyucu kullanan bir kullanıcı olarak, alt fon satırlarını da fare kullanıcısı kadar etkin biçimde gezebilmek istiyorum, böylece tıklanabilir olanları açabileyim ve devre dışı olanların sebebini öğrenebileyim.

#### Acceptance Criteria

1. THE Fund_Holdings_Panel SHALL tıklanabilir Fund_Position_Row'lara klavye odağı alabilen bir DOM özniteliği (`tabindex="0"` veya semantik link/buton) atar.
2. THE Fund_Holdings_Panel SHALL devre dışı Fund_Position_Row'lar için `aria-disabled="true"` özniteliğini ayarlar AND klavye `Tab` sırasına dahil etmez.
3. THE Fund_Holdings_Panel SHALL tıklanabilir bir Fund_Position_Row için ekran okuyucularda "fon detayını aç" anlamına gelen erişilebilir bir etiket sağlar (örn. `aria-label`).
4. WHEN bir Fund_Position_Row klavye ile odaklanır, THE Fund_Holdings_Panel SHALL görünür bir odak halkası (focus ring) gösterir.
5. THE Fund_Holdings_Panel SHALL devre dışı bir satırın tooltip mesajını programatik olarak (`aria-describedby` veya eşdeğer) odaklandığında ekran okuyuculara sunar.

### Requirement 8: Eksik alan tespit edildiğinde Restricted_Data_Notice gösterimi

**User Story:** Bir kullanıcı olarak, bir alt fonun detay sayfasını açtığımda bazı verilerin eksik olabileceğini önceden anlamak istiyorum, böylece eksik alanları bir hata olarak yorumlamayayım.

#### Acceptance Criteria

1. WHEN Fund_Detail_Page Requirement 1'deki akışla bir alt fon kodu için açılır AND yüklenen fund detail yanıtında en az bir kritik alan (ör. son fiyat, son güncelleme tarihi) tanımsız veya boş, THE Fund_Detail_Page SHALL Restricted_Data_Notice'i sayfa başında gösterir.
2. THE Restricted_Data_Notice SHALL kullanıcıya "Bu fon için kısıtlı veri mevcut" anlamına gelen kısa, tek satırlık bir mesaj gösterir AND hangi alanın eksik olduğunu listelemek zorunda değildir.
3. WHEN Fund_Detail_Page tüm kritik alanlarını dolu olarak yükler, THE Fund_Detail_Page SHALL Restricted_Data_Notice'i göstermez.
4. WHEN kullanıcı Restricted_Data_Notice üzerindeki kapatma kontrolüne tıklar, THE Fund_Detail_Page SHALL bu uyarıyı yalnızca mevcut görüntüleme oturumu için gizler.

### Requirement 9: Response cache geçişi

**User Story:** Bir backend operatörü olarak, yeni bayrak alanı yayına alınırken eski cache yanıtlarının kendiliğinden geçersiz hâle gelmesini istiyorum, böylece deploy sonrası elle bir cache temizliğine ihtiyaç kalmasın.

#### Acceptance Criteria

1. WHEN Fund_Service `TEFAS_Tradable_Flag` desteğini ekleyen değişiklik yayına alınır, THE Backend SHALL Holdings_Response_Schema_Version değerini bir artırır.
2. WHEN Holdings_Response_Schema_Version değeri artırılır, THE Backend SHALL artırılmadan önce yazılmış holdings response cache girdilerinin yeni şema sürümüne uyumlu olmadığında okunmadan yenilenmesini sağlar.
3. THE Backend SHALL aynı değişiklik kapsamında KAP_Holdings_Parse_Version değerini değiştirmez ve KAP holdings parse cache girdilerini yeniden oluşturmaya zorlamaz.
