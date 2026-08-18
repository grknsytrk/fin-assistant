# Interview Story (STAR) - Fin-Rag

## Durum

BIST şirketleri, KAP finansalları ve TEFAS fon verileri farklı kaynaklarda ve farklı güncellik kurallarıyla bulunuyordu. Kullanıcıların tek ekranda karşılaştırılabilir finansal veriye ulaşması gerekiyordu.

## Görev

React + FastAPI uygulamasında KAP ve TEFAS veri akışlarını birleştirmek, cache davranışını güvenilir hale getirmek ve finansal ekranların aynı veri sözleşmesini kullanmasını sağlamak.

## Aksiyon

- KAP snapshot'larını şirket kimliği ve cache durumu üzerinden yönettim.
- Fon listesi, geçmişi ve portföy dağılımı için TEFAS adapter/cache akışını kullandım.
- Finansal tablo ve overview ekranlarında boş/yanlış sınıflandırılmış metriklerin filtrelenmesini düzelttim.
- Üretim akışını sadeleştirmek için PDF/RAG/embedding katmanını kaldırdım; eski hisse route'larını Genel Bakış'a yönlendirdim.

## Sonuç

Uygulama yalnızca aktif veri kaynakları ve yapılandırılmış endpoint'lerle çalışıyor. Fon, KAP ve finansal tablo ekranları aynı backend sözleşmelerini kullanıyor; gereksiz model ve index bağımlılıkları bulunmuyor.
