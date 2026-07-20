## Self Attention
### Self Attention là gì ?
* Là một cơ chế cho phép mỗi token trong một chuỗi xem xét mọi token khác trong cùng chuỗi đó, bao gồm cả chính nó, để hiểu ngữ cảnh
* Từ "Attention" có nghĩa là mỗi token tự quyết định mức độ tập trung của mình vào các token khác
* Vì vậy, mỗi từ trong câu đều xem xét mọi từ khác trong cùng câu đó và tìm ra những từ nào quan trọng nhất đối với ý nghĩa của nó
* Ví dụ: Câu : "The animal did not cross the street because it was too tired."
  * Vậy câu hỏi đặt ra là: từ này `it` ám chỉ điều gì?
  * Ta thừa biết nó ám chỉ `the animal`
  * Vậy còn câu: "The animal did not cross the street because it was too wide."
  * Cùng 1 cấu trúc câu nhưng `it` ở 2 câu lại có ý nghĩa hoàn toàn khác nhau
  * Não bộ của chúng ta làm thế nào để hiểu được 2 ngữ cảnh đó ? Chúng ta sẽ xem xét các từ xung quanh từ `it`, từ đó, từ `tired` thu hút sự chú ý của chúng ta về phía từ `the animal`, còn `wide` thu hút chúng ta về từ `the street`
  * Đây chính xác là những gì xảy ra bên trong LLMs, mỗi từ đều xem xét mọi từ khác trong cùng 1 câu để hiểu nghĩa của nó. Mô hình này dành nhiều sự chú ý hơn cho những từ quan trọng và ít chú ý hơn với những từ không quan trọng
  * Nếu thiếu đi sự "Self-Attention" trong 1 câu, từ ngữ `it` sẽ chỉ là một từ chung chung, không có ý nghĩa thực sự

#### Các vector Query, Key, Value
* Với mỗi token, self-attention cần trả lời 2 câu hỏi:
  * Những từ khóa nào khác trong câu này quan trọng đối với tôi?
  * Tôi nên tập trung vào từng khía cạnh đến mức độ nào?
* Để trả lời những câu hỏi này, chúng ta tạo ra ba vectơ cho mỗi mã thông báo đầu vào:
  * **Query vector**: token này đang tìm kiếm điều gì.
  * **Key vector**: những gì mã thông báo này cung cấp cho người khác.
  * **Value vector**: thông tin thực tế mà mã thông báo này mang theo.
* Từ "Self" trong cụm từ "Self Attention" xuất phát từ một sự thật rất quan trọng: 
  ```text
    Truy vấn (Q), Khóa (K) và Giá trị (V) đều xuất phát từ CÙNG một chuỗi đầu vào.
  ```
* Đây chính là điều làm nên sự "Tự" chú ý. Cùng một chuỗi tạo ra các câu hỏi, cùng một chuỗi tạo ra các đáp án để khớp với những câu hỏi đó, và cùng một chuỗi cung cấp các giá trị. Chuỗi đó đang tự chú ý đến chính nó.
* Nói một cách đơn giản, nếu câu đầu vào của chúng ta là "Con vật không băng qua đường", thì Q, K và V đều được suy ra từ chính câu "Con vật không băng qua đường".

### Step-by-step xây dựng 1 cơ chế Self Attention
* **Bước 1**: Bắt đầu với các vector embedding (nhúng) đầu vào. Mỗi token trong câu được chuyển đổi thành một vector. (output1)</br></br>
* **Bước 2**: Tạo ra ba vectơ cho mỗi token bằng cách nhân ma trận nhúng (output1) với ba ma trận trọng số khác nhau:
  * Q = $X * W_Q$
  * K = $X * W_K$
  * V = $X * W_V$
  * Ở đây, W_Q, W_K, và W_Vlà các ma trận trọng số mà mô hình học được trong quá trình huấn luyện.</br></br>
* **Bước 3**: Tính tích vô hướng của Qvới ma trận chuyển vị của K. Đây là phép nhân ma trận cho ra điểm số chú ý. Chúng ta sử dụng ma trận chuyển vị để các hình dạng được căn chỉnh chính xác cho phép nhân.
  * scaled_score = $\frac{Q.K^T}{\sqrt{d_k}}$ (nhân vô hướng thì cho ra 1 số)
  * Điểm số cho ta biết mức độ tương tác giữa hai token. Điểm số càng cao nghĩa là sự trùng khớp càng mạnh giữa truy vấn của token này và khóa của token kia 
  * Việc điều chỉnh tỷ lệ này được thực hiện để giữ cho các số nằm trong phạm vi ổn định, tránh việc hàm softmax tạo ra các giá trị cực đoan. Nếu không có sự điều chỉnh tỷ lệ này, độ dốc có thể trở nên rất nhỏ trong quá trình huấn luyện</br></br>
* **Bước 4**: Áp dụng hàm softmax lên các điểm số đã được hiệu chỉnh. Điều này chuyển đổi các điểm số thành xác suất. Tổng của mỗi hàng trong ma trận lúc này bằng 1.
  * Hàm softmax: $ \hat{y}_j = \text{softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=0}^{1} e^{z_k}} $
  * Attentions Weights = softmax(scaled_score) = $softmax(\frac{Q.K^T}{\sqrt{d_k}})$
  * Các trọng số này cho chúng ta biết, đối với mỗi token, nó nên dành bao nhiêu sự chú ý cho mỗi token khác.</br></br>
* **Bước 5**: Nhân các `attention weights` với ma trận Giá trị V. Điều này cho ra kết quả đầu ra cuối cùng.
  * output = $softmax(\frac{Q.K^T}{\sqrt{d_k}}) * V$
  * Kết quả đầu ra là một biểu diễn mới của mỗi token, trong đó ý nghĩa của token được làm phong phú thêm bởi ngữ cảnh của các token khác

### Vì sao Self-Attention lại hiệu quả đến vậy
* Trước khi có cơ chế Tự Chú Ý (Self Attention), các mô hình như RNN và LSTM xử lý từng token một trong một chuỗi. Điều này rất chậm và làm suy yếu mối liên kết giữa các từ cách xa nhau. Cơ chế Tự Chú Ý đã giải quyết cả hai vấn đề này.
  * Song song hóa: Tất cả các token được xử lý cùng một lúc. Chúng ta không cần phải chờ một token hoàn thành trước khi xử lý token tiếp theo. Điều này giúp quá trình huấn luyện trên GPU diễn ra rất nhanh.
  * Phụ thuộc tầm xa: Mỗi token có thể trực tiếp tham chiếu đến mọi token khác, ngay cả khi chúng cách nhau rất xa trong câu. Không có sự suy giảm theo khoảng cách.
  * Hiểu theo ngữ cảnh: Mỗi từ vựng sẽ có ý nghĩa phong phú hơn dựa trên các từ xung quanh. Cùng một từ có thể có cách diễn đạt khác nhau trong các câu khác nhau tùy thuộc vào ngữ cảnh.
  * Tính linh hoạt: Cơ chế Tự chú ý không quan tâm đến thứ tự tính toán. Nó có thể xử lý các chuỗi dài và nắm bắt các mối quan hệ phức tạp.


## Toán học đằng sau Attention - Q,K,V
### Công thức của Attention
* Ta có công thức sau: $${\displaystyle \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$
* Ở đây:
  * Q là ma trận truy vấn (Query)
  * K là ma trận khóa (Key)
  * V là ma trận giá trị (Value)
  * $K^T$ là ma trận chuyển vị của K
  * $d_k$ là chiều (dim) của vector khóa (Key)</br></br>
* Ý tưởng rất đơn giản: chúng ta so sánh những gì mỗi từ đang tìm kiếm ( Truy vấn ) với những gì mà mọi từ khác cung cấp ( Từ khóa ), và sau đó sử dụng những so sánh đó để thu thập thông tin thực tế ( Giá trị ) từ những từ có liên quan nhất

### Thiết lập: From Words to Vector
* Giả sử lấy ví dụ 1 câu đơn giản: "Tôi yêu AI"
* Trong Transformer , mỗi từ trước tiên được chuyển đổi thành một vectơ (tức là một danh sách các số) được gọi là embedding . Để dễ hiểu hơn, chúng ta sẽ sử dụng các vectơ rất nhỏ có kích thước 4 (ví dụ: d_emb = 4)
    ```text
    "I"    → [1.0, 0.0, 1.0, 0.0]
    "love" → [0.0, 1.0, 0.0, 1.0]
    "AI"   → [1.0, 1.0, 0.0, 0.0]
    ```
* Xếp chồng chúng thành một ma trận đầu vào $X$ có dạng `3 x 4` (3 từ, mỗi từ có 4 số)
    ```
    X = | 1.0  0.0  1.0  0.0 |   ← "I"
        | 0.0  1.0  0.0  1.0 |   ← "love"
        | 1.0  1.0  0.0  0.0 |   ← "AI"
    ```
  
* Ở đây, mỗi hàng là một từ và mỗi cột là một số trong biểu diễn embedding. Đây là điểm xuất phát của chúng ta.

## Multi-head Attention
* Trong lý thuyết Self-Attention, chúng ta chỉ có một bộ Q, K và V. Nhưng một bộ là không đủ để nắm bắt tất cả các loại mối quan hệ khác nhau trong một câu.
* Trong cơ chế Tự chú ý đa đầu (Multi-Head Self Attention - MHSA), chúng ta chạy cơ chế Tự chú ý nhiều lần song song, mỗi lần chạy có một tập hợp các tham số W_Q, W_K, và riêng W_V. Mỗi lần chạy song song được gọi là một "đầu" (head). Mỗi đầu học cách tập trung vào một khía cạnh khác nhau của câu.
* Sau khi tất cả các đầu ra đã hoàn thành công việc của mình, chúng ta sẽ nối chúng lại với nhau và truyền qua thêm một lớp tuyến tính nữa. Điều này mang lại cho chúng ta kết quả cuối cùng của cơ chế MHSA (Multi-Head Self Attention)
* MHSA = nhiều quá trình Self-Attention chạy song song
* Bằng cách này, mô hình có thể nắm bắt nhiều loại mối quan hệ khác nhau trong cùng một câu tại cùng một thời điểm.