#!/bin/bash
# Parallel Project Gutenberg downloader
# Usage: ./data.sh [target_size_mb] [output_file] [num_processes]

TARGET_SIZE_MB=${1:-300}
OUTPUT_FILE=${2:-"gutenberg_corpus.txt"}
NUM_PROCESSES=${3:-4}
TARGET_SIZE_BYTES=$((TARGET_SIZE_MB * 1024 * 1024))

# Create temporary directory for process coordination
TEMP_DIR=$(mktemp -d)
LOCK_FILE="$TEMP_DIR/write.lock"
SIZE_FILE="$TEMP_DIR/current_size"
BOOK_COUNT_FILE="$TEMP_DIR/book_count"

# Initialize coordination files
echo "0" > "$SIZE_FILE"
echo "0" > "$BOOK_COUNT_FILE"
> "$OUTPUT_FILE"

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR"
    # Kill all background processes
    jobs -p | xargs -r kill 2>/dev/null
}
trap cleanup EXIT

echo "Downloading Project Gutenberg books to reach ${TARGET_SIZE_MB}MB using $NUM_PROCESSES processes..."

download_book() {
    local id=$1
    local process_id=$2
    
    # Download content
    local content=$(curl -s --max-time 8 "https://www.gutenberg.org/files/$id/$id-0.txt")
    
    # Basic validation
    if [ -z "$content" ] || echo "$content" | head -5 | grep -qi "<html\|<!doctype"; then
        return 1
    fi
    
    if [ ${#content} -lt 1000 ]; then
        return 1
    fi
    
    # Extract title and author
    local title=$(echo "$content" | grep -i "^Title:" | head -1 | sed 's/^[Tt]itle: *//; s/\r//g' | tr -d '\n\r' | head -c 50)
    local author=$(echo "$content" | grep -i "^Author:" | head -1 | sed 's/^[Aa]uthor: *//; s/\r//g' | tr -d '\n\r' | head -c 30)
    
    # Fallback title
    [ -z "$title" ] && title="Book_$id"
    
    # Write to temporary file first
    local temp_file="$TEMP_DIR/book_${process_id}_${id}.txt"
    echo "$content" > "$temp_file"
    
    # Thread-safe append to main file using flock
    (
        flock -x 200
        cat "$temp_file" >> "$OUTPUT_FILE"
        
        # Update size and count
        local current_size=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || wc -c < "$OUTPUT_FILE")
        local current_books=$(cat "$BOOK_COUNT_FILE")
        echo "$current_size" > "$SIZE_FILE"
        echo "$((current_books + 1))" > "$BOOK_COUNT_FILE"
        
        # Report progress
        local current_mb=$((current_size / 1024 / 1024))
        if [ -n "$author" ]; then
            echo "P$process_id Book $id: \"$title\" by $author → ${current_mb}MB / ${TARGET_SIZE_MB}MB ($((current_books + 1)) books)"
        else
            echo "P$process_id Book $id: \"$title\" → ${current_mb}MB / ${TARGET_SIZE_MB}MB ($((current_books + 1)) books)"
        fi
        
    ) 200>"$LOCK_FILE"
    
    # Clean up temp file
    rm -f "$temp_file"
    
    return 0
}

worker_process() {
    local process_id=$1
    local start_id=$2
    local step=$3
    
    local book_id=$start_id
    local books_downloaded=0
    
    while [ $book_id -le 50000 ]; do
        # Check if target size reached
        local current_size=$(cat "$SIZE_FILE")
        if [ $current_size -ge $TARGET_SIZE_BYTES ]; then
            echo "P$process_id: Target size reached, stopping"
            break
        fi
        
        if download_book $book_id $process_id; then
            books_downloaded=$((books_downloaded + 1))
        else
            echo "P$process_id Book $book_id: skip"
        fi
        
        book_id=$((book_id + step))
        sleep 0.1
    done
    
    echo "P$process_id: Downloaded $books_downloaded books"
}

# Start worker processes with non-overlapping ranges
for i in $(seq 1 $NUM_PROCESSES); do
    worker_process $i $i $NUM_PROCESSES &
done

# Wait for all processes to complete
wait

# Final summary
final_size=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || wc -c < "$OUTPUT_FILE")
final_books=$(cat "$BOOK_COUNT_FILE")
echo
echo "Complete: $final_books books, $(echo $final_size | numfmt --to=iec)B"